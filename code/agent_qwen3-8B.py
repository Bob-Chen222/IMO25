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

# --- CONFIGURATION ---
# The model to use. "gemini-1.5-flash" is fast and capable.
#MODEL_NAME = "gemini-1.5-flash-latest" 
# MODEL_NAME = "gemini-2.5-pro" 
SOLVER_MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
SAFE_MODEL_NAME = SOLVER_MODEL_NAME.replace("/", "-")
VERIFIER_MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
SOLVER_NAME = ""

_log_file = None
original_print = print

def log_print(*args, **kwargs):
    return
    """
    Custom print function that writes to both stdout and log file.
    """
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Add timestamp to lines starting with ">>>>>"
    if message.startswith('>>>>>'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {message}"
    
    # Print to stdout
    original_print(message)
    
    # Also write to log file if specified
    if _log_file is not None:
        _log_file.write(message + '\n')
        _log_file.flush()  # Ensure immediate writing

# Replace the built-in print function
# print = log_print

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
Below is the bug report. If you agree with certain items in it, improve your program so that it becomes complete, correct, and rigorous. 
Note that the evaluator who generates the bug report may misunderstand your solution and thus make mistakes. 
If you do not agree with certain items in the bug report, add detailed comments or explanations in your code to avoid such misunderstanding. 
Your revised solution must strictly follow the instructions in the system prompt, including wrapping the full program in ```<code>``` and ensuring every part is justified and well-documented.
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


verification_reminder = """
### Verification Task Reminder ###

Your task is to act as an expert code reviewer. Generate only the **Summary**. 
State the final verdict and list all Bugs with their code location, description, and the index of a failing input. 
If no Bugs exist, explicitly state “No Bugs identified.”
Always return one shared oracle program with a JSON test suite of inputs (covering normal, boundary, and edge cases). 
If Bugs exist, ensure the test suite includes inputs that expose them.
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
    Convert your Gemini-style payload into a generic chat message list:
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

def _messages_to_prompt_with_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use tokenizer chat template if available.
    """
    # Convert to HF chat format: [{"role": "...", "content": "..."}]
    # HF expects "role" in {"system","user","assistant"} and "content" string.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: crude concatenation
    lines = []
    for m in messages:
        if m["role"] == "system":
            lines.append(f"<|system|>\n{m['content']}\n")
        elif m["role"] == "assistant":
            lines.append(f"<|assistant|>\n{m['content']}\n")
        else:
            lines.append(f"<|user|>\n{m['content']}\n")
    lines.append("<|assistant|>\n")  # cue the model to respond
    return "\n".join(lines)

def serve(payloads: List[Dict[str, Any]], model_name: str, max_new_tokens: int = 16384, batch_size=10) -> str:
    """
    Generate a chat response locally using Hugging Face Transformers.
    - Respects temperature/topP from payload["generationConfig"] when present.
    - Uses the model's chat template if available.
    """
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

    # Load model + tokenizer

    # Prepare messages
    prompts_batched = []
    for i in range(0, (len(payloads) + batch_size - 1) // batch_size, batch_size):
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

            output_array = response_data['output']
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
            print(json.dumps(response_data, indent=2))
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
    if(verbose):
        print(">>>>>>> Start verification.")
    p2s = build_request_payloads(system_prompt=verification_system_prompt, 
        question_prompts=newsts
        )
    
    assert len(p2s) == len(problem_statements)
    
    if(verbose):
        print(">>>>>>> Verification prompt:")
        print(json.dumps(p2s, indent=4))

    ress = serve(p2s, VERIFIER_MODEL_NAME)

    outs = extract_text_from_responses(ress) 
    print("outs: ", outs)

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(outs, indent=4))

    check_correctness_list = ["""
    Respond only with "yes" or "no". Does the solution meet the problem requirements and produce correct results for all valid inputs?
    """   + "\n\n" + out for out in outs]
    prompts = build_request_payloads(system_prompt="", question_prompts=check_correctness_list)
    rs = serve(prompts, VERIFIER_MODEL_NAME)
    os = extract_text_from_responses(rs)

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(os, indent=4))

    bug_reports = []
    os = [o.strip() for o in os]
    bug_reports = extract_detailed_solutions(outs, "Summary", False)
    
    return bug_reports, os
        


def init_explorations(problem_statements, verbose=False, other_prompts=[]):
    p1s  = build_request_payloads(
            system_prompt=step1_prompt,
            question_prompts=problem_statements,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    # print(f">>>>>> Initial prompt.")
    # print(json.dumps(p1s, indent=4))
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

    # print(f">>>>>>> First solution: ") 
    # print(json.dumps(output1s, indent=4))

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
    solutions = extract_text_from_responses(response2s)
    assert len(solutions) == len(p1s)
    # print(f">>>>>>> Corrected solution: ")
    # print(json.dumps(solutions, indent=4))
    
    print(f">>>>>>> Verify the solution.")
    verifys, good_verifys = verify_solutions(problem_statements, solutions, verbose)

    # print(f">>>>>>> Initial verification: ")
    # print(json.dumps(verifys, indent=4))
    # print(f">>>>>>> verify results: {good_verifys}")
    
    return p1, solutions, verifys, good_verifys

def agent(problem_statements, other_prompts=[], memory_file=None, resume_from_memory=False):
    
    # Start fresh
    current_iteration = 0
    solution = None
    verify = None
    
    p1s, solution, verifys, good_verifys = init_explorations(problem_statements, False, other_prompts)
    if(solution is None):
        print(">>>>>>> Failed in finding a complete solution.")
        return None

    # we will just do the first round and see if it's good enough for now
    for i, (verify, good_verify) in enumerate(zip(verifys, good_verifys)):
        result = 1 if "yes" in good_verify.lower() else 0
        obj = {
            "idx": i,
            "problem_statement": problem_statement,
            "solution": solution,
            "verify": verify,
            "result": result
        }
        with open(f"final_res_" + SAFE_MODEL_NAME + ".jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj) + '\n')
        
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt', 
                       help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    parser.add_argument('--memory', '-mem', type=str, help='Path to memory file for saving/loading state (optional)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from memory file if provided')
    parser.add_argument('--receipt', '-rec', default='receipt.jsonl', help='Path to receipt file')
    
    args = parser.parse_args()

    max_runs = args.max_runs
    memory_file = args.memory
    resume_from_memory = args.resume
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)
    
    if memory_file:
        print(f"Memory file: {memory_file}")
        if resume_from_memory:
            print("Resume mode: Will attempt to load from memory file")

    # Set up logging if log file is specified
    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    
    
    # problem_statement = read_file_content(args.problem_file)

    # read receipt file and upload all the problem statement as a list
    question_ids = []
    dataset, subset, split = "", "", ""
    with open (args.receipt, 'r', encoding='utf-8') as f:
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

    agent(problem_statements, other_prompts, memory_file, resume_from_memory)
    
    # Close log file if it was opened
    close_log_file()
