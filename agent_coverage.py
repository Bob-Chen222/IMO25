import os
from pickle import FALSE
import sys
import json
from textwrap import indent
import requests
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from vllm import LLM, SamplingParams
import openai
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio
import tempfile
import coverage

# This file is as same as agent.py but will use test coverage as a metric to evaluate the solutions

# --- CONFIGURATION ---
# The model to use. "gemini-1.5-flash" is fast and capable.
#MODEL_NAME = "gemini-1.5-flash-latest" 
# MODEL_NAME = "gemini-2.5-pro" 
SOLVER_MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
SAFE_MODEL_NAME = SOLVER_MODEL_NAME.replace("/", "-")
VERIFIER_MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "gpt-5"
SAFE_VERIFIER_MODEL_NAME = VERIFIER_MODEL_NAME.replace("/", "-")
SOLVER_NAME = ""

_log_file = None
original_print = print

def run_coverage_on_inputs(program: str, inputs: List[str]) -> Dict[str, Any]:
    """
    Execute PROGRAM (a Python source string) once per input (fed to stdin)
    and compute statement coverage using coverage.py.

    Returns:
      {
        "coverage_percent": int,      # e.g., 87
        "coverage_rate": float,       # e.g., 0.87
        "covered_lines": List[int],
        "missing_lines": List[int],
        "missing_content": Dict[int, str],  # lineno -> exact source
        "executable_lines": List[int],
        "filename": str,
      }
    """
    if not inputs:
        return {
            "coverage_percent": 0,
            "coverage_rate": 0.0,
            "covered_lines": [],
            "missing_lines": [],
            "missing_content": {},
            "executable_lines": [],
            "filename": "",
        }

    # Write the program to a temp file so coverage can track it
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        program_path = f.name

    cov = coverage.Coverage()
    cov.start()

    # Compile once (faster), exec per input with fresh globals
    with open(program_path, "rb") as f:
        code = compile(f.read(), program_path, "exec")

    original_stdin = sys.stdin
    try:
        for inp in inputs:
            sys.stdin = io.StringIO(inp)
            glb = {"__name__": "__main__", "__file__": program_path}
            try:
                exec(code, glb)
            except BaseException:
                # Don't let a failing case nuke the run; still count coverage so far.
                pass
    finally:
        sys.stdin = original_stdin

    cov.stop()
    cov.save()

    # Use analysis2 for a simple dict-like result
    a2 = cov.analysis2(program_path)
    stmts: List[int] = a2["statements"]
    missing: List[int] = a2["missing"]
    executed = sorted(set(stmts) - set(missing))

    # Load exact source lines for missing-content map
    missing_content: Dict[int, str] = {}
    if missing:
        with open(program_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ln in missing:
            if 1 <= ln <= len(lines):
                missing_content[ln] = lines[ln - 1].rstrip("\n")

    denom = len(stmts)
    rate = (len(executed) / denom) if denom else 0.0

    return {
        "coverage_rate": rate,
        "covered_lines": executed,
        "missing_lines": missing,
        "missing_content": missing_content,
    }



async def _request_one(
    sem: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    messages: Dict[str, Any],
    n: int,
    idx: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    async with sem:
        resp = await client.chat.completions.create(
            model=VERIFIER_MODEL_NAME,
            messages=messages,
            n=n,
            temperature=temperature,
        )
        answers = [choice.message.content for choice in resp.choices]
        return {"id": idx, "responses": answers}


async def batch_openai_request(
    data: List[Dict[str, Any]],
    n: int = 1,
    max_concurrency: int = 10,
    temperature: float = 0.1,
    top_p: float = 1.0,
) -> List[Dict[str, List[str]]]:
    sem = asyncio.Semaphore(max_concurrency)

    tasks = [
        _request_one(sem, client, item, n, item["idx"], temperature) for item in data
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Requesting", total=len(tasks))

    await client.close()
    return results


def set_log_file(log_file_path):
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

# Global variables for logging

step1_prompt = """
### Core Instructions ###

*   **Correctness is Paramount:** Your primary goal is to produce a valid, executable Python program that fully solves the given problem. Every line of code must be logically sound and consistent with the problem requirements.
*   **Honesty About Correctness:** If you cannot provide a fully correct solution, do not write code that only looks plausible but contains hidden flaws. Instead, output only significant partial progress that you can rigorously justify. Examples of significant partial progress include:
    *   Implementing a correct helper function or core algorithmic step.
    *   Handling one or more cases correctly within a case-based solution.
    *   Providing a correct algorithmic skeleton with stubs where the missing parts are clearly marked.

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings.

**a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution** just state “Complete.”
    *   **For a partial solution:** use a few sentences to briefly explain high-level idea

**2. Detailed Solution**

*   Always output only the program (complete or partial) inside the code fence ```<code>``` (no other text outside the code fence).
*   If partial, include clear # TODO: markers where work remains. No speculative stubs—only code that runs or is well-justified. Keep comments tied to the high-level idea.



### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every line of code is correct, executable, and consistent with the described strategy.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your program carefully. Correct any coding errors, fix logic mistakes, and fill in missing justifications or comments if any. 
If the solution is incomplete, extend it rigorously or clearly mark unimplemented parts with TODOs. 
Your revised output must strictly follow the instructions in the system prompt, including wrapping the full program inside ```<code>``` and ensuring every function, loop, and conditional is well-documented and tied back to the method sketch.
"""

check_verification_prompt = """
Can you carefully review each item in your list of identified issues or findings about the program? 
Are they valid or overly strict? An expert reviewer must be able to distinguish between a genuine coding flaw and a concise implementation that is nonetheless correct, 
and to correct their own assessment when necessary.

If you feel that modifications to any item or its justification are necessary, please produce a revised list. 
In your final output, please directly start with **Summary** (no need to justify the new list separately).
"""

correction_prompt = """
You are given a bug report about a Python program that attempts to solve a specific problem
and you are asked to review the bug report and the program and make necessary corrections to the program.

Your task has two required parts:

**Part 1 — Bug Report & Test Case Review**  
- Identify what in the bug report is correct and genuinely points to necessary improvements.  
- Identify what in the bug report is incorrect or based on misunderstanding, and explain why it is wrong.  
- Examine the test cases provided in the bug report:  
  - Determine which test cases are valid (i.e., they correctly capture edge cases or expected behavior from the problem statement).  
  - Point out any invalid or misleading test cases and explain why they do not apply.  
- Point out any issue or bug in the program provided in the bug report
- Always clarify your reasoning to avoid future misunderstanding.   

**Part 2 — Updated Program**  
- Apply the valid improvements identified in your bug report review.  
- For issues you rejected, keep your original logic unchanged 
- Ensure the program is complete, correct, rigorous, and well-documented.  
- Wrap the entire program in the following format:  
  ```<code>
  <full runnable program here>
  </code>  

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Bug Report Review**
<Write your review here.>
- Cover: correct items, incorrect items (and why), and validity of each provided test case.
- Suggestions for what can be further improved regarding the test suites generated.

**2. Detailed Solution**
*   Always output only the program (complete or partial) inside the code fence ```<code>``` (no other text outside the code fence).
*   If partial, include clear # TODO: markers where work remains. No speculative stubs—only code that runs or is well-justified. Keep comments tied to the high-level idea.
"""

verification_system_prompt = """
You are an expert programmer and meticulous code reviewer for high-stakes algorithmic tasks. Your primary duty is to rigorously verify the provided Python program against the stated problem. A submission is judged correct **only if it is logically sound, fully executable, and produces correct results for all valid inputs**. Code that reaches the right answer via flawed logic, missing cases, undefined behavior, or luck is **incorrect or incomplete**.

### Instructions ###

**0. Scope & Role**
- Your sole goal is to determine whether the submitted program is functionally correct for the stated problem. 
- Ignore style, comments, and code justification; only correctness matters.

**1. Bug Identification**
- A **Bug** is any defect that makes the program produce incorrect results for at least one valid input.  
  - This includes logic errors, unhandled cases, wrong boundaries, invalid assumptions, or complexity issues that cause the program to fail under stated limits.  
  - Bugs may be identified either by reasoning about the code or by showing a mismatch between the program and a correct oracle on specific inputs.

**2. Oracle & Test Suite**
- Provide a **single shared oracle program**: a minimal, clearly correct reference implementation.  
- Provide a set of **independent test inputs** that expose the discovered bugs.  
- If the program is judged correct, the test suite should still cover normal, boundary, and edge cases to confirm overall robustness.
- Use the following JSON format:

{
  "program": "<entire runnable oracle program as a single string (no backticks, no markdown)>",
  "test_inputs": [
    { "idx": 0, "input_string": "<exact input 0>" },
    { "idx": 1, "input_string": "<exact input 1>" }
  ]
}

**3. Output Format**
Your response must contain exactly one section: **Summary**.

**Summary**
- **Final Verdict**: One sentence stating overall correctness, e.g. “The program is correct.” / “The program is incorrect.”  
- **List of Findings**: Bullet every issue discovered. For each:
  - **Location**: Quote or pinpoint the relevant code or description.
  - **Issue**: Concise description (always classified as **Bug**).
  - **Exposing Input**: Reference the index of the test input from the shared oracle suite that demonstrates the bug.

**4. Rigor Rules**
- No hand-waving: every claim must be tied to the submitted code, the problem specification, or the oracle's defined behavior.
- Always use precise language about inputs, outputs, and edge cases.
- If the problem specification is ambiguous, state the ambiguity and verify under the most standard interpretation(s).

"""

verification_correction_prompt = '''
You are given feedback on your previous bug report and an updated program based on it.  
If you agree with the feedback, update your bug report to make it correct and rigorous.  
If you disagree, keep your original logic.  
Always follow the system prompt instructions in your final solution.  
'''

verification_reminder = """
### Verification Task Reminder ###

Your task is to act as an expert code reviewer. Generate only the **Summary**. 
State the final verdict and list all Bugs with their code location, description, and the index of a failing input. 
If no Bugs exist, explicitly state “No Bugs identified.”
Always return one shared oracle program with a JSON test suite of inputs (covering normal, boundary, and edge cases). 
If Bugs exist, ensure the test suite includes inputs that expose them.
"""

test_coverage_improvement_prompt = """
 ### Improvement Task ###
            Please review the original solution and the coverage report above. 
            Identify what you can do to improve the coverage,
            and make necessary changes to the inputs in your test suite to improve the coverage.
            Your revised output must strictly follow the instructions in the system prompt, and only change the test suite inputs 
            but not the oracle program.
"""

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def build_request_payloads(system_prompt, question_prompts, other_prompts=None):
    """
    Builds the JSON payload for the Gemini API request, using the
    recommended multi-turn format to include a system prompt.
    """

    payloads = []
    for question_prompt in question_prompts:
        
        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [
                {
                    "text": system_prompt 
                }
                ]
            },
        "contents": [
            {
            "role": "user",
            "parts": [{"text": question_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 1.0,
            "thinkingConfig": { "thinkingBudget": 32768} 
        },
        }

        if other_prompts:
            for prompt in other_prompts:
                payload["contents"].append({
                    "role": "user",
                    "parts": [{"text": prompt}]
                })
        payloads.append(payload)

    return payloads

def _payload_to_message(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert your payloads into a conversation
    [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    messages: List[Dict[str, str]] = []
    # system
    sys = payload.get("systemInstruction", {}).get("parts", [])
    if sys and isinstance(sys, list):
        sys_text = " ".join(p.get("text", "") for p in sys if isinstance(p, dict))
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text.strip()})

    # conversation
    for turn in payload.get("contents", []):
        role = turn.get("role", "user")
        parts = turn.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if text.strip():
            # Map unknown roles to user/assistant conservatively
            if role not in ("system", "user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": text.strip()})

    return messages

def serve(payloads: List[Dict[str, Any]], model_name: str, max_new_tokens: int = 32768, batch_size=5) -> str:
    """
    Generate a chat response either locally or via API
    - Respects temperature/topP from payload["generationConfig"] when present.
    - Uses the model's chat template if available.
    """
    
    if "gpt" in model_name.lower():
        # Use OpenAI API
        # Convert payloads to OpenAI format
        openai_payloads = []
        for idx, payload in enumerate(payloads):
            messages = _payload_to_message(payload)
            openai_payloads.append({
                "idx": idx,
                "prompt": messages,
            })
        # Call OpenAI API asynchronously
        responses = asyncio.run(batch_openai_request(openai_payloads))
        # Extract texts
        all_texts = []
        # sort responses by idx and also only keep the responses field
        responses = [x["response"] for x in sorted(responses, key=lambda x: x["id"])]
        for resp in responses:
            text = extract_text_from_responses(resp, model_name)
            all_texts.append(text)
        return all_texts

    if len(payloads) < batch_size:
        batch_size = len(payloads)
    # Parse generation config
    gen_cfg = payloads[0].get("generationConfig", {}) or {}
    temperature: float = float(gen_cfg.get("temperature", 0.1))
    top_p: float = float(gen_cfg.get("topP", 1.0))

    llm = LLM(
        model=model_name,   # same HF model name or local path
        trust_remote_code=True,
    )

    # Define generation params (temperature, top_p, max tokens, etc.)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    ) 

    # Prepare messages
    prompts_batched = []
    for i in range(0, (len(payloads) + batch_size - 1) // batch_size):
        payloads_prepared = payloads[i * batch_size: (i + 1) * batch_size]
        prompts = []
        for payload in payloads_prepared:
            message = _payload_to_message(payload)
            prompts.append(message)
        prompts_batched.append(prompts)
    
    assert len(prompts_batched) != 0



    all_texts = []
    for batch in prompts_batched:
        outputs = llm.chat(batch, sampling_params, use_tqdm=True)
        for out in outputs:
            # take the first candidate (index=0)
            # strip anything behind the first </think>
            marker = "</think>"
            # do the split
            try:
                split_text = out.outputs[0].text.split(marker, 1)[1]
                all_texts.append(split_text)
            except IndexError as e:
                all_texts.append(out.outputs[0].text[:500])
             
    return all_texts

def extract_text_from_responses(responses_data, model_name="") -> str:
    """
    Extracts the generated text from the API response JSON.
    Handles potential errors if the response format is unexpected.
    """

    if not "gpt" in model_name.lower():
        return responses_data
    else:
        assert "gpt" in model_name.lower()
        try:
        # The output is an array, we need to find the message with text content

            output_array = responses_data['output']
            for item in output_array:
                if item['type'] == 'message' and 'content' in item:
                    content_array = item['content']
                    for content_item in content_array:
                        if content_item['type'] == 'output_text':
                            return content_item['text']
            
            # Fallback: if no text found, return empty string
            return ""
        except (KeyError, IndexError, TypeError) as e:
            print("Error: Could not extract text from the API response.")
            print(f"Reason: {e}")
            print("Full API Response:")
            print(json.dumps(responses_data, indent=2))
            #sys.exit(1)
            raise e  

def extract_detailed_solutions(solutions, marker='```', after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    detailed_solutions = [] 
    for solution in solutions:
        idx = solution.find(marker)
        if idx == -1:
            detailed_solutions.append(solution.strip())
            continue
        if(after):
            detailed_solutions.append(solution[idx + len(marker):].strip())
        else:
            detailed_solutions.append(solution[:idx].strip())
    return detailed_solutions

def verify_solutions(problem_statements, solutions, verbose=False):
    assert len(problem_statements) != 0

    dsols = extract_detailed_solutions(solutions)
    newsts = []
    for problem_statement, dsol in zip(problem_statements, dsols):
        newst = f"""

        ======================================================================
        ### Problem ###

        {problem_statement}

        ======================================================================
        ### Solution ###

        {dsol}

        {verification_reminder}
        """
        newsts.append(newst)

    print(">>>>>>> Start verification.")

    p2s = build_request_payloads(system_prompt=verification_system_prompt, 
        question_prompts=newsts
        )
    
    assert len(p2s) == len(problem_statements)

    outs = serve(p2s, VERIFIER_MODEL_NAME)

    check_correctness_list = ["""
    Respond only with "yes" or "no". Does the solution meet the problem requirements and produce correct results for all valid inputs?
    """   + "\n\n" + out for out in outs]
    prompts = build_request_payloads(system_prompt="", question_prompts=check_correctness_list)
    decisions = serve(prompts, VERIFIER_MODEL_NAME)

    bug_reports = []
    decisions_final = [d.strip() for d in decisions]
    bug_reports = extract_detailed_solutions(outs, "Summary", False)
    print(">>>>>>> Verification done.")
    
    return bug_reports, decisions_final
        


def init_explorations(problem_statements, verbose=False, other_prompts=[]):
    p1s  = build_request_payloads(
            system_prompt=step1_prompt,
            question_prompts=problem_statements,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    response1s = []
    if os.path.exists("response1s_" + SAFE_MODEL_NAME + ".jsonl"):
        print(">>>>>>> Found existing response1s.jsonl, loading...")
        with open("response1s_" + SAFE_MODEL_NAME + ".jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                response1s.append(obj['response'])
    else:
        response1s = serve(p1s, SOLVER_MODEL_NAME)
        with open("response1s_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
            for idx, (p1, response1) in enumerate(zip(p1s, response1s)):
                obj = {
                    "idx": idx,
                    "prompt": p1,
                    "response": response1
                }
                f.write(json.dumps(obj) + '\n')

    output1s = extract_text_from_responses(response1s)

    print(f">>>>>>> Self improvement start:")
    for p1 , output1 in zip(p1s, output1s):
        p1["contents"].append(
            {"role": "model",
            "parts": [{"text": output1}]
            }
        )
        p1["contents"].append(
            {"role": "user",
            "parts": [{"text": self_improvement_prompt}]
            }
        )
    assert len(p1s) != 0
    if os.path.exists("response2s_" + SAFE_MODEL_NAME + ".jsonl"):
        response2s = []
        with open("response2s_" + SAFE_MODEL_NAME + ".jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                response2s.append(obj['response'])
    else:
        response2s = serve(p1s, SOLVER_MODEL_NAME)
        # save response2s to jsonl file
        with open("response2s_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
            for idx, (p1, response2) in enumerate(zip(p1s, response2s)):
                obj = {
                    "idx": idx,
                    "prompt": p1,
                    "response": response2
                }
                f.write(json.dumps(obj) + '\n')

    print(f">>>>>>> Self improvement done.")
    assert len(response2s) == len(p1s)
    solutions = response2s
    assert len(solutions) == len(p1s)
    
    verifys, good_verifys = verify_solutions(problem_statements, solutions, verbose)
    
    return p1, solutions, verifys, good_verifys

def second_round_explorations(solutions, verifys, good_verifys, verbose=False):
    '''
    Given solutions, previous result of verification, run the program against the test
    cases and find the coverage
    '''

    # extract the input lists from verifys
    input_list_list = []
    for verify in verifys:
        try:
            verify_json = json.loads(verify)
            test_inputs = verify_json.get("test_inputs", [])
            input_list = [test_input.get("input_string", "") for test_input in test_inputs]
            input_list_list.append(input_list)
        except json.JSONDecodeError:
            input_list_list.append([])
    
    coverage_list = [0] * len(solutions)
    for solution, input_list in zip(solutions, input_list_list):
        dsol = extract_detailed_solutions([solution])[0]
        cov_result = run_coverage_on_inputs(dsol, input_list)
        coverage_list.append(cov_result)

    # for each solution, we are going to keep track the coverage and if it no longer increases or we have already ran 2 times, we stop
    max_iterations = 2
    current_iteration = 0
    while current_iteration < max_iterations:
        print(f">>>>>>> Second round exploration iteration {current_iteration + 1}")
        p3s = []
        for solution, coverage in zip(solutions, coverage_list):
            coverage_percent = coverage.get("coverage_rate", 0.0) * 100
            p3 = f"""
            ======================================================================
            ### Original Solution ###

            {solution}

            ======================================================================
            ### Coverage Report ###

            The current test suite achieves {coverage_percent:.2f}% statement coverage.
            The following lines are not covered: {coverage.get("missing_lines", [])}
            The following lines are covered: {coverage.get("covered_lines", [])}
            The following lines are executable: {coverage.get("executable_lines", [])}
            The following lines have missing content: {coverage.get("missing_content", {})}

            ======================================================================
            """
        
        p3_payloads = build_request_payloads(
                system_prompt=correction_prompt,
                question_prompts=[]
            )

        for verify, entry in zip(verifys, p3_payloads):
            entry["contents"].append({
                "role": "assistant",
                "parts": [{"text": verify}]
            })
            entry["contents"].append({
                "role": "user",
                "parts": [{"text": test_coverage_improvement_prompt}]  
            })
            

        if os.path.exists("response3s_" + SAFE_MODEL_NAME + ".jsonl"):
            response3s = []
            with open("response3s_" + SAFE_MODEL_NAME + ".jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    response3s.append(obj['response'])
        else:
            response3s = serve(p3_payloads, SOLVER_MODEL_NAME)
            # save response3s to jsonl file
            with open("response3s_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
                for idx, (p3, response3) in enumerate(zip(p3_payloads, response3s)):
                    obj = {
                        "idx": idx,
                        "prompt": p3,
                        "response": response3
                    }
                    f.write(json.dumps(obj) + '\n')

        
        check_correctness_list = ["""
        Respond only with "yes" or "no". Does the solution meet the problem requirements and produce correct results for all valid inputs?
        """   + "\n\n" + out for out in response3s]
        prompts = build_request_payloads(system_prompt="", question_prompts=check_correctness_list)
        decisions = serve(prompts, VERIFIER_MODEL_NAME)

        bug_reports = []
        decisions_final = [d.strip() for d in decisions]
        bug_reports = extract_detailed_solutions(response3s, "Summary", False)
        print(">>>>>>> Verification done.")

        return solutions, bug_reports, decisions_final

        
    




    




def agent(problem_statements, other_prompts=[], memory_file=None, resume_from_memory=False):
    
    # Start fresh
    current_iteration = 0
    solutions = []
    verifys = []
    good_verifys = []


    
    if os.path.exists(SAFE_VERIFIER_MODEL_NAME + "/" + "final_res_" + SAFE_MODEL_NAME + ".jsonl"):
        print(">>>>>>> Found existing final_res.jsonl, loading...")
        with open(SAFE_VERIFIER_MODEL_NAME + "/" + "final_res_" + SAFE_MODEL_NAME + ".jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                solutions.append(obj['solution'])
                verifys.append(obj['verify'])
    else:   
        p1s, solutions, verifys, good_verifys = init_explorations(problem_statements, False, other_prompts)

        for i, (verify, good_verify) in enumerate(zip(verifys, good_verifys)):
            result = 1 if "yes" in good_verify.lower() else 0
            obj = {
                "idx": i,
                "problem_statement": problem_statement,
                "solution": solutions[i],
                "verify": verify,
                "result": result
            }
            with open(SAFE_VERIFIER_MODEL_NAME + "/" + f"final_res_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj) + '\n')


    if os.path.exists(SAFE_VERIFIER_MODEL_NAME + "/" + "final_res_2_" + SAFE_MODEL_NAME + ".jsonl"):
        print(">>>>>>> Found existing final_res_2.jsonl, loading...")
        with open(SAFE_VERIFIER_MODEL_NAME + "/" + "final_res_2_" + SAFE_MODEL_NAME + ".jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                solutions.append(obj['solution'])
                verifys.append(obj['verify'])
    else:
    # start the second round explorations
        print(">>>>>>> Start second round explorations...")
        solutions2, bug_reports, good_verifys = second_round_explorations(problem_statements, solutions, verifys, good_verifys)
        for i, (verify, good_verify) in enumerate(zip(bug_reports, good_verifys)):
            result = 1 if "yes" in good_verify.lower() else 0
            obj = {
                "idx": i,
                "problem_statement": problem_statement,
                "solution": solutions2[i],
                "verify": verify,
                "result": result
            }
            with open(SAFE_VERIFIER_MODEL_NAME + "/" + f"final_res_2_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj) + '\n')

        
if __name__ == "__main__":
    # TODO: these variables are just placeholder, CHANGE them if needed in the future
    max_runs = 0
    memory_file = None
    resume_from_memory = None
    receipt = "receipt.jsonl"

    # read receipt file and upload all the problem statement as a list
    question_ids = []
    dataset, subset, split = "", "", ""
    with open (receipt, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            question_ids.append(obj['question_id'])
    

    # load dataset and only keep the ones that are in question_ids
    ds = load_dataset("microsoft/rStar-Coder", "seed_sft", split="train")
    filtered_ds = ds.filter(lambda example: example['question_id'] in question_ids)
    print(f"Loaded {len(filtered_ds)} problems from dataset {dataset}, subset {subset}, split {split}")

    problem_statements = []
    for question_id in question_ids:
        matched = filtered_ds.filter(lambda example: example['question_id'] == question_id)
        if(len(matched) == 0):
            print(f"Warning: question_id {question_id} not found in dataset")
            continue
        if(len(matched) > 1):
            print(f"Warning: question_id {question_id} has multiple entries in dataset, using the first one")
        problem_statement = matched[0]['question']
        problem_statements.append(problem_statement)
    print(f"Loaded {len(problem_statements)} problem statements.")

    agent(problem_statements, [], memory_file, resume_from_memory)
    
    # Close log file if it was opened
    close_log_file()
