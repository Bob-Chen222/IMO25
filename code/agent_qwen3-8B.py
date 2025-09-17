import os
from pickle import FALSE
import sys
import json
from textwrap import indent
import requests
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from type import List, Dict, Any, Optional
import torch

# --- CONFIGURATION ---
# The model to use. "gemini-1.5-flash" is fast and capable.
#MODEL_NAME = "gemini-1.5-flash-latest" 
# MODEL_NAME = "gemini-2.5-pro" 
MODEL_NAME = "Qwen/Qwen3-8B"

def save_memory(memory_file, problem_statement, other_prompts, current_iteration, max_runs, solution=None, verify=None):
    """
    Save the current state to a memory file.
    """
    memory = {
        "problem_statement": problem_statement,
        "other_prompts": other_prompts,
        "current_iteration": current_iteration,
        "max_runs": max_runs,
        "solution": solution,
        "verify": verify,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"Memory saved to {memory_file}")
        return True
    except Exception as e:
        print(f"Error saving memory to {memory_file}: {e}")
        return False

def load_memory(memory_file):
    """
    Load the state from a memory file.
    """
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        print(f"Memory loaded from {memory_file}")
        return memory
    except Exception as e:
        print(f"Error loading memory from {memory_file}: {e}")
        return None

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

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., “I have successfully solved the problem. The final program is correct and fully functional.”
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to implement, e.g., “I have not found a complete solution, but I have rigorously implemented the core dynamic programming routine that computes optimal substructure values.”
*   **b. Method Sketch:** Present a high-level, conceptual outline of your coding solution. This sketch should allow an expert programmer to understand the logical flow of your algorithm without reading the full code. It should include:
    *   A narrative of your overall algorithmic strategy.
    *   The full and precise description of any key subroutines or data structures.
    *   If applicable, describe how you decomposed the problem (e.g., main loop, helper functions, case splits).
    *   The entire program must be wrapped as follows: ```<code> ```.

**2. Detailed Solution**

*   **Present the full Python program. Each part of the code must be logically justified and well-documented with comments. The level of detail should be sufficient for an expert to verify correctness without needing to guess your intentions.
    *Every function, loop, and conditional should have a clear role tied back to the method sketch.
    *If the solution is partial, clearly mark unimplemented sections with placeholders (e.g., # TODO) and explain what remains.
    *No speculative, unverified code is allowed.



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
- **Final Verdict**: One sentence stating overall correctness, e.g. “The program is correct.” / “The program is invalid due to Bugs.”  
- **List of Findings**: Bullet every issue discovered. For each:
  - **Location**: Quote or pinpoint the relevant code or description.
  - **Issue**: Concise description (always classified as **Bug**).
  - **Exposing Input**: Reference the index of the test input from the shared oracle suite that demonstrates the bug.

**4. Rigor Rules**
- No hand-waving: every claim must be tied to the submitted code, the problem specification, or the oracle’s defined behavior.
- Always use precise language about inputs, outputs, and edge cases.
- If the problem specification is ambiguous, state the ambiguity and verify under the most standard interpretation(s).

**Example Summary Format (illustrative)**

Final Verdict: The program is **invalid** due to Bugs in empty input and single-element handling.

List of Findings:
- Location: `if not arr: return 1`  
  - Issue: **Bug** — The specification defines the empty-input minimum as 0; returning 1 is incorrect.  
  - Exposed by test input index `0`.

- Location: `for i in range(1, n):`  
  - Issue: **Bug** — Skips the case `n = 1`, leading to incorrect behavior.  
  - Exposed by test input index `1`.

**Shared Oracle + Test Suite (JSON format)**

{
  "program": "<entire runnable oracle program as a single string (no backticks, no markdown)>",
  "test_inputs": [
    { "idx": 0, "input_string": "<exact input 0>" },
    { "idx": 1, "input_string": "<exact input 1>" }
  ]
}
"""


verification_reminder = """
### Verification Task Reminder ###

Your task is to act as an expert code reviewer. Generate only the **Summary**. 
State the final verdict and list all Bugs with their code location, description, and the index of a failing input. 
If Bugs exist, also return one shared oracle program with a JSON test suite of inputs exposing them.
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

def build_request_payload(system_prompt, question_prompt, other_prompts=None):
    """
    Builds the JSON payload for the Gemini API request, using the
    recommended multi-turn format to include a system prompt.
    """
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

    return payload

def _payload_to_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
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

def serve_huggingface(payload: Dict[str, Any], model_name: str = MODEL_NAME, max_new_tokens: int = 4096) -> str:
    """
    Generate a chat response locally using Hugging Face Transformers.
    - Respects temperature/topP from payload["generationConfig"] when present.
    - Uses the model's chat template if available.
    """
    # Parse generation config
    gen_cfg = payload.get("generationConfig", {}) or {}
    temperature: float = float(gen_cfg.get("temperature", 0.1))
    top_p: float = float(gen_cfg.get("topP", 1.0))

    # Prepare messages
    messages = _payload_to_messages(payload)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build prompt
    prompt = _messages_to_prompt_with_template(tokenizer, messages)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Sampling flags (temperature<=0 → greedy)
    do_sample = temperature is not None and float(temperature) > 0.0

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(1e-6, float(temperature)) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Slice off the prompt to get only new tokens
    gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

def extract_text_from_response(response_data):
    """
    Extracts the generated text from the API response JSON.
    Handles potential errors if the response format is unexpected.
    """
    try:
        return response_data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError, TypeError) as e:
        print("Error: Could not extract text from the API response.")
        print(f"Reason: {e}")
        print("Full API Response:")
        print(json.dumps(response_data, indent=2))
        #sys.exit(1)
        raise e 

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if(after):
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):

    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_reminder}
"""
    if(verbose):
        print(">>>>>>> Start verification.")
    p2 = build_request_payload(system_prompt=verification_system_prompt, 
        question_prompt=newst
        )
    
    if(verbose):
        print(">>>>>>> Verification prompt:")
        print(json.dumps(p2, indent=4))

    res = serve_huggingface(p2, MODEL_NAME, 4096)
    out = extract_text_from_response(res) 

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    prompt = build_request_payload(system_prompt="", question_prompt=check_correctness)
    r = serve_huggingface(prompt, MODEL_NAME, 4096)
    o = extract_text_from_response(r) 

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(o, indent=4))
        
    bug_report = ""

    if("yes" not in o.lower()):
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

        """p2["contents"].append(
            {"role": "model",
            "parts": [{"text": bug_report}]
            }
        )
        p2["contents"].append(
            {"role": "user",
            "parts": [{"text": check_verification_prompt}]
            }
        )

        if(verbose):
            print(">>>>>>> Review bug report prompt:")
            print(json.dumps(p2["contents"][-2:], indent=4))

        res = send_api_request(get_api_key(), p2)
        out = extract_text_from_response(res) 
    """

    if(verbose):
        print(">>>>>>>Bug report:")
        print(json.dumps(bug_report, indent=4))
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """

    p1 = build_request_payload(system_prompt="",    question_prompt=check_complete_prompt)
    r = serve_huggingface(p1, MODEL_NAME, 4096)
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()


def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    p1  = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(p1, indent=4))

    response1 = serve_huggingface(p1, MODEL_NAME, 4096)
    output1 = extract_text_from_response(response1)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
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

    response2 = serve_huggingface(p1, MODEL_NAME, 4096)
    solution = extract_text_from_response(response2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    #print(f">>>>>>> Check if solution is complete:"  )
    #is_complete = check_if_solution_claimed_complete(output1)
    #if not is_complete:
    #    print(f">>>>>>> Solution is not complete. Failed.")
    #    return None, None, None, None
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(f">>>>>>> Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f">>>>>>> verify results: {good_verify}")
    
    return p1, solution, verify, good_verify

def agent(problem_statement, other_prompts=[], memory_file=None, resume_from_memory=False):
    if resume_from_memory and memory_file:
        # Load memory and resume from previous state
        memory = load_memory(memory_file)
        if memory:
            problem_statement = memory.get("problem_statement", problem_statement)
            other_prompts = memory.get("other_prompts", other_prompts)
            current_iteration = memory.get("current_iteration", 0)
            solution = memory.get("solution", None)
            verify = memory.get("verify", None)
            print(f"Resuming from iteration {current_iteration}")
        else:
            print("Failed to load memory, starting fresh")
            current_iteration = 0
            solution = None
            verify = None
    else:
        # Start fresh
        current_iteration = 0
        solution = None
        verify = None
    
    if solution is None:
        p1, solution, verify, good_verify = init_explorations(problem_statement, True, other_prompts)
        if(solution is None):
            print(">>>>>>> Failed in finding a complete solution.")
            return None
    else:
        # We have a solution from memory, need to get good_verify
        _, good_verify = verify_solution(problem_statement, solution)

    error_count = 0
    correct_count = 1
    success = False
    for i in range(current_iteration, 30):
        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        if("yes" not in good_verify.lower()):
            # clear
            correct_count = 0
            error_count += 1

            #self improvement
            print(">>>>>>> Verification does not pass, correcting ...")
            # establish a new prompt that contains the solution and the verification

            p1 = build_request_payload(
                system_prompt=step1_prompt,
                question_prompt=problem_statement,
                #other_prompts=["You may use analytic geometry to solve the problem."]
                other_prompts=other_prompts
            )

            p1["contents"].append(
                {"role": "model",
                "parts": [{"text": solution}]
                }
            )
            
            p1["contents"].append(
                {"role": "user",
                "parts": [{"text": correction_prompt},
                          {"text": verify}]
                }
            )

            print(">>>>>>> New prompt:")
            print(json.dumps(p1, indent=4))
            response2 = serve_huggingface(p1, MODEL_NAME, 4096)
            solution = extract_text_from_response(response2)

            print(">>>>>>> Corrected solution:")
            print(json.dumps(solution, indent=4))


            #print(f">>>>>>> Check if solution is complete:"  )
            #is_complete = check_if_solution_claimed_complete(solution)
            #if not is_complete:
            #    print(f">>>>>>> Solution is not complete. Failed.")
            #    return None

        print(f">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if("yes" in good_verify.lower()):
            print(">>>>>>> Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0
 

        # Save memory every iteration
        if memory_file:
            save_memory(memory_file, problem_statement, other_prompts, i, 30, solution, verify)
        
        if(correct_count >= 5):
            print(">>>>>>> Correct solution found.")
            print(json.dumps(solution, indent=4))
            return solution

        elif(error_count >= 10):
            print(">>>>>>> Failed in finding a correct solution.")
            # Save final state before returning
            if memory_file:
                save_memory(memory_file, problem_statement, other_prompts, i, 30, solution, verify)
            return None

    if(not success):
        print(">>>>>>> Failed in finding a correct solution.")
        # Save final state before returning
        if memory_file:
            save_memory(memory_file, problem_statement, other_prompts, 30, 30, solution, verify)
        return None
        
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
    
    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts, memory_file, resume_from_memory)
            if(sol is not None):
                print(f">>>>>>> Found a correct solution in run {i}.")
                print(json.dumps(sol, indent=4))
                break
        except Exception as e:
            print(f">>>>>>> Error in run {i}: {e}")
            continue
    
    # Close log file if it was opened
    close_log_file()
