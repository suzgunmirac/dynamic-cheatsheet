import argparse 
import ast 
import json
import openai
import random 
import re 
import time 
import os
import numpy as np
from metrics import qa_score 
import datetime

# Add detailed logging functionality
def log_llm_call(log_dir, call_info):
    """
    Log detailed information about each LLM call
    
    Args:
        log_dir: Directory to save logs
        call_info: Dictionary containing call information
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename = f"{call_info['role']}_{call_info['call_id']}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Add timestamp to call_info
    call_info['timestamp'] = timestamp
    call_info['datetime'] = datetime.datetime.now().isoformat()
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(call_info, f, indent=2, ensure_ascii=False)
    
    print(f"[LOG] {call_info['role']} call logged to {filename}")

def timed_llm_call(client, model, prompt, role, call_id, log_dir=None):
    """
    Make an LLM call with detailed timing and logging
    
    Args:
        client: OpenAI client
        model: Model name
        prompt: Input prompt
        role: Role of the LLM (generator/reflector/curator)
        call_id: Unique identifier for this call
        log_dir: Directory to save logs (optional)
    
    Returns:
        Tuple of (response_content, call_info)
    """
    start_time = time.time()
    prompt_time = time.time()
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        response_time = time.time()
        total_time = response_time - start_time
        
        response_content = response.choices[0].message.content
        
        # Create detailed call info
        call_info = {
            "role": role,
            "call_id": call_id,
            "model": model,
            "prompt": prompt,
            "response": response_content,
            "prompt_time": prompt_time - start_time,  # Time to prepare prompt
            "response_time": response_time - prompt_time,  # Time to get response
            "total_time": total_time,
            "prompt_length": len(prompt),
            "response_length": len(response_content),
        }
        
        print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
        
        # Log if directory provided
        if log_dir:
            log_llm_call(log_dir, call_info)
        
        return response_content, call_info
    
    except Exception as e:
        error_time = time.time()
        call_info = {
            "role": role,
            "call_id": call_id,
            "model": model,
            "prompt": prompt,
            "error": str(e),
            "total_time": error_time - start_time,
            "prompt_length": len(prompt),
        }
        
        print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
        
        if log_dir:
            log_llm_call(log_dir, call_info)
        
        raise e

from utils_claude import *
# Import evaluation functions from run_benchmark_finlora.py to ensure comparable results
import sys
sys.path.append('.')
try:
    from run_benchmark_finlora import evaluate_accuracy, process_batched
except ImportError:
    print("Warning: Could not import evaluation functions from run_benchmark_finlora.py")
    # Keep our local versions as fallback


#### API key information ####

# SambaNova (different keys for different models)
sambanova_generator_api_key = "59d1e878-392e-4bf1-b476-86f066577728"
sambanova_reflector_api_key = "ba502f4f-a797-40b6-8e99-13d54651520d"
sambanova_curator_api_key = "3f4abfeb-79a6-47d6-99d8-c08d21431db2"
sambanova_base_url = "https://api.sambanova.ai/v1"

# Together (optional)
together_api_key = "d5365c77f1ed57d2266be68a10463bb05cb2eb6996555c50c0b649b12b5755e9"
together_base_url = "https://api.together.xyz/v1"

###---------------------####

# Delta curator prompt for bulletpoint approach
delta_curator_prompt = """You are a master curator of financial knowledge. Your job is to identify what new insights should be added to an existing cheatsheet based on a reflection from a failed attempt.

**Instructions:**
- Review the existing cheatsheet and the reflection from the failed attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are missing from the current cheatsheet
- Do NOT regenerate the entire cheatsheet - only provide the additions needed
- Format your response as a JSON object with specific sections
- If no new insights are needed for a section, omit that section from the JSON
- Be concise and specific - each addition should be actionable

**Existing Cheatsheet:**
{}

**Reflection (identifying what went wrong):**
{}

**Question Context:**
{}

**Your Task:**
Provide ONLY the new bulletpoints that should be added to the cheatsheet. Format as JSON:

```json
{{
    "financial_strategies": "New strategy or insight to add to financial strategies section",
    "formulas_calculations": "New formula or calculation method to add",
    "common_mistakes": "New common mistake to avoid, e.g., 'When doing X, forgot to add -1 to the last index'",
    "problem_solving_heuristics": "New rule of thumb or heuristic to add",
    "context_clues": "New indicator or clue to help identify problem approach",
    "code_snippets": "New code template or snippet to add",
    "others": "Any other important insight that doesn't fit in the above categories"
}}
```

**Delta JSON:**
"""


def evaluate_accuracy(out, target, target_type_list):
    """
    Evaluate accuracy based on the same logic as run_benchmark_finlora.py
    """
    correct_count = 0
    response = []

    target_type_list_lower = [str(t).lower() for t in target_type_list]

    if len(out) != len(target):
        raise ValueError("Input lists 'out' and 'target' must have the same length.")

    for x, y in zip(out, target):
        # Ensure inputs are strings and convert to lowercase
        x_str = str(x)
        y_str = str(y)
        x_lower = x_str.lower()
        y_lower = y_str.lower()

        found_labels_info = []

        # Find the first occurrence of each valid label in the output x
        for valid_label in target_type_list_lower:
            try:
                # string.find() returns -1 if not found, or the starting index
                index = x_lower.find(valid_label)
                if index != -1:
                    found_labels_info.append({'label': valid_label, 'index': index})
            except AttributeError:
                print(f"Warning: Attribute error during find for x='{x_str}', label='{valid_label}'")
                continue

        is_current_prediction_correct = False

        if not found_labels_info:
            is_current_prediction_correct = False
        else:
            found_labels_info.sort(key=lambda item: item['index'])

            # The first label in the sorted list is the one that appeared earliest.
            first_occurred_label = found_labels_info[0]['label']

            # Check if this first occurred label matches the target label y_lower.
            if first_occurred_label == y_lower:
                is_current_prediction_correct = True
            else:
                is_current_prediction_correct = False

        # Update correct count and the response list
        if is_current_prediction_correct:
            correct_count += 1
            response.append(y)  # Append the original target label (y)
        else:
            response.append(x)  # Append the original LLM output (x)

    accuracy = 0.0
    if len(out) > 0:
        accuracy = correct_count / len(out)

    return accuracy, response


def process_batched(out_text_list, target_list):
    """
    Process batched outputs for XBRL finer tasks
    """
    processed_out_text_list = []
    processed_target_list = []

    for out_text, target in zip(out_text_list, target_list):
        split_output = [x.strip().replace("\n", "") for x in out_text.split(',')]  # Split and strip whitespace
        split_target = [x.strip().replace("\n", "") for x in target.split(',')]  # Split and strip whitespace
        processed_target_list += (split_target)  # Keep the split target
        output_len = len(split_output)
        target_len = len(split_target)

        if output_len != target_len:
            if output_len > target_len:
                # Output is longer, truncate
                processed_out_text_list += (split_output[:target_len])
            else:
                # Target is longer, pad output with empty strings
                padding_needed = target_len - output_len
                processed_out_text_list += (split_output + [""] * padding_needed)
        else:
            # Lengths match, use output as is
            processed_out_text_list += (split_output)

    return processed_out_text_list, processed_target_list


def extract_cheatsheet(
    response: str,
    old_cheatsheet: str,
) -> str:
    """
    Extracts the cheatsheet from the model response.

    Arguments:
        response : str : The response from the model.
        old_cheatsheet : str : The old cheatsheet to return if the new one is not found.
    Returns:
        str : The extracted cheatsheet (if not found, returns the old cheatsheet).
    """
    response = response.strip()
    # <cheatsheet> (content) </cheatsheet>
    if "<cheatsheet>" in response:
        try:
            txt = response.split("<cheatsheet>")[1].strip()
            txt = txt.split("</cheatsheet>")[0].strip()
            return txt
        except:
            return old_cheatsheet
    else:
        return old_cheatsheet


def extract_delta_json(response: str) -> dict:
    """
    Extracts the delta JSON from the model response.
    
    Arguments:
        response : str : The response from the model.
        
    Returns:
        dict : The extracted delta JSON (empty dict if not found).
    """
    try:
        # Try to find JSON block
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            # Try to find JSON directly
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end].strip()
        
        delta_dict = json.loads(json_str)
        return delta_dict
    except:
        print("Failed to extract delta JSON, returning empty dict")
        return {}


def merge_delta_to_cheatsheet(old_cheatsheet: str, delta_dict: dict) -> str:
    """
    Merges delta additions into the existing cheatsheet.
    
    Arguments:
        old_cheatsheet : str : The existing cheatsheet.
        delta_dict : dict : The delta additions to merge.
        
    Returns:
        str : The updated cheatsheet.
    """
    if not delta_dict or old_cheatsheet == "(empty)":
        # Initialize empty cheatsheet if needed
        if old_cheatsheet == "(empty)":
            old_cheatsheet = """## FINANCIAL STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    updated_cheatsheet = old_cheatsheet
    
    # Section mapping
    section_mapping = {
        "financial_strategies": "## FINANCIAL STRATEGIES & INSIGHTS",
        "formulas_calculations": "## FORMULAS & CALCULATIONS", 
        "code_snippets": "## CODE SNIPPETS & TEMPLATES",
        "common_mistakes": "## COMMON MISTAKES TO AVOID",
        "problem_solving_heuristics": "## PROBLEM-SOLVING HEURISTICS",
        "context_clues": "## CONTEXT CLUES & INDICATORS",
        "others": "## OTHERS"
    }
    
    for key, value in delta_dict.items():
        if key in section_mapping:
            # Handle both string and list values
            if isinstance(value, list):
                # If it's a list, join all items
                content_items = [item.strip() for item in value if item and str(item).strip()]
                if not content_items:
                    continue
            elif isinstance(value, str):
                # If it's a string, use it directly
                if not value.strip():
                    continue
                content_items = [value.strip()]
            else:
                # Skip unknown types
                print(f"Warning: Unknown value type for {key}: {type(value)}")
                continue
            
            section_header = section_mapping[key]
            
            # Find the section in the cheatsheet
            if section_header in updated_cheatsheet:
                # Find the end of this section (next ## or end of document)
                section_start = updated_cheatsheet.find(section_header)
                section_content_start = section_start + len(section_header)
                
                # Find next section or end of document
                next_section = updated_cheatsheet.find("\n##", section_content_start)
                if next_section == -1:
                    next_section = len(updated_cheatsheet)
                
                # Extract current section content
                current_section = updated_cheatsheet[section_content_start:next_section]
                
                # Add new bulletpoints (one for each item)
                new_bulletpoints = ""
                for item in content_items:
                    new_bulletpoints += f"\n- {item}"
                
                updated_section = current_section + new_bulletpoints
                
                # Replace in the cheatsheet
                updated_cheatsheet = (updated_cheatsheet[:section_content_start] + 
                                    updated_section + 
                                    updated_cheatsheet[next_section:])
            else:
                # Section doesn't exist, add it at the end
                new_section_content = ""
                for item in content_items:
                    new_section_content += f"\n- {item}"
                updated_cheatsheet += f"\n\n{section_header}{new_section_content}"
    
    return updated_cheatsheet


def extract_answer(
    response: str,
) -> str:
    """
    Extracts the final answer from the model response.

    Arguments:
        response : str : The response from the model.

    Returns:
        str : The extracted final answer (if not found, returns "No final answer found").
    """
    matches = re.findall(r"Finish\[(.*?)\]", response)

    if matches:
        last_answer = matches[-1]
        return last_answer
    else:
        return "No final answer found"


def parse_context_and_question(all_context):
    """
    Parse context to extract question and context parts.
    Handles both training and test data formats, and both batched and breakdown versions.
    
    Returns:
        tuple: (context, question)
    """
    # Check for specific question formats first (regardless of whether it starts with "You are XBRL expert")
    
    # Batched version (4 questions) - most specific match first
    if "Answer the following 4 independent questions by providing only  4 US GAAP tags answers in the order of the questions. Each answer must be saperated by a comma (,).  Provide nothing else." in all_context:
        index = all_context.index("Answer the following 4 independent questions by providing only  4 US GAAP tags answers in the order of the questions. Each answer must be saperated by a comma (,).  Provide nothing else.")
        context, question = all_context[:index], all_context[index:]
        return context, question
    
    # Breakdown version (single question)  
    elif "Answer the following question by providing only the US GAAP tag and nothing else." in all_context:
        index = all_context.index("Answer the following question by providing only the US GAAP tag and nothing else.")
        context, question = all_context[:index], all_context[index:]
        return context, question
    
    # Legacy test data format (for backward compatibility)
    elif "Answer the following 4 independent questions by providing only" in all_context:
        index = all_context.index("Answer the following 4 independent questions by providing only")
        context, question = all_context[:index], all_context[index:]
        return context, question
    
    # Training data format fallback (starts with "You are XBRL expert" but no recognizable question pattern)
    elif "You are XBRL expert" in all_context:
        # For training data, the entire context IS the question
        # We'll treat the GAAP tags list as context and everything else as question
        if "Here is a list of US GAAP tags options:" in all_context:
            try:
                # Split at a reasonable point
                parts = all_context.split("Here is a list of US GAAP tags options:")
                if len(parts) >= 2:
                    # First part + tag list = context, the rest = question  
                    context_part = parts[0] + "Here is a list of US GAAP tags options:" + parts[1].split(".")[0] + "."
                    question_part = ".".join(parts[1].split(".")[1:])
                    return context_part.strip(), question_part.strip()
            except Exception as e:
                print(f"Error parsing training data: {e}")
                # Fallback: treat everything as question
                return "", all_context
        
        # Fallback: treat everything as question, empty context
        return "", all_context
    
    # Unknown format fallback
    else:
        print(f"Warning: Unknown data format, treating entire context as question")
        print(f"Context preview: {all_context[:200]}...")
        return "", all_context


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--test_dataset_path", type=str, default="data/finlora/test/finer_test_broken_down.jsonl", help="Path to test dataset for evaluation")
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--num_test_samples", default=100, type=int, help="Number of test samples to evaluate each epoch")
    parser.add_argument("--curator_model", type=str, default="Llama-4-Maverick-17B-128E-Instruct")
    parser.add_argument("--reflector_model", type=str, default="Llama-4-Maverick-17B-128E-Instruct")
    parser.add_argument("--generator_model", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--use_together_api", action="store_true", help="Use Together API instead of SambaNova")
    args = parser.parse_args()
    return args 


def initialize_clients(use_together=False):
    """
    Initialize separate clients for generator, reflector, and curator
    
    Returns:
        tuple: (generator_client, reflector_client, curator_client)
    """
    if use_together:
        # TOGETHER client (same API key for all)
        generator_client = openai.OpenAI(api_key=together_api_key, base_url=together_base_url)
        reflector_client = openai.OpenAI(api_key=together_api_key, base_url=together_base_url)
        curator_client = openai.OpenAI(api_key=together_api_key, base_url=together_base_url)
        print("Using Together API for all models")
    else:
        # SAMBANOVA client (different API keys)
        generator_client = openai.OpenAI(api_key=sambanova_generator_api_key, base_url=sambanova_base_url)
        reflector_client = openai.OpenAI(api_key=sambanova_reflector_api_key, base_url=sambanova_base_url)
        curator_client = openai.OpenAI(api_key=sambanova_curator_api_key, base_url=sambanova_base_url)
        print("Using SambaNova API with separate keys:")
        print(f"  Generator key: {sambanova_generator_api_key}")
        print(f"  Reflector key: {sambanova_reflector_api_key}")
        print(f"  Curator key: {sambanova_curator_api_key}")
    
    return generator_client, reflector_client, curator_client


def get_sleep_time(use_together=False):
    """Get appropriate sleep time based on API provider"""
    return 2 if use_together else 60


def relaxed_check_xbrl(final_answer, gt_answer):
    pred = final_answer.split(",")
    label = gt_answer.split(",")
    label = [val.lower().strip() for val in label]
    count = 0 
    for prediction in pred: 
        prediction  = prediction.lower().strip()
        if prediction in label: count += 1
    score = count/len(pred)
    if score > 0.5: return True 
    return False


def relaxed_check(final_answer, gt_answer):
    # used for financebench 
    score = qa_score(final_answer, gt_answer)
    if score > 0.4: return True 
    return False 


def evaluate_single_answer(final_answer, gt_answer):
    """
    Evaluate a single answer and return if it's correct
    """
    return relaxed_check_xbrl(final_answer, gt_answer)


def evaluate_test_set(generator_client, generator_model, cheatsheet, test_samples, num_samples=100, log_dir=None):
    """
    Evaluate the current cheatsheet on test data
    
    Returns:
        dict: evaluation results
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING ON TEST SET ({num_samples} samples)")
    print(f"{'='*40}")
    
    # Limit test samples
    if len(test_samples) > num_samples:
        # Use random sampling for fair evaluation
        random.seed(42)  # Fixed seed for reproducible evaluation
        test_samples = random.sample(test_samples, num_samples)
    
    test_correct = 0
    test_total = 0
    test_answers = []
    test_targets = []
    
    for i, sample in enumerate(test_samples):
        try:
            task_dict = ast.literal_eval(sample)
            all_context = task_dict["context"]
            gt_answer = task_dict["target"]
            
            # Parse context using same logic as training
            # XBRL finer 
            index = all_context.index("Answer the following 4 independent questions by providing only")
            context, question = all_context[:index], all_context[index:]

            # Generate answer (no reflection in test mode)
            gen_prompt = generator_prompt.format(cheatsheet, "(empty)", question, context)
            
            # Use timed LLM call with logging
            call_id = f"test_eval_{i}"
            gen_response, call_info = timed_llm_call(
                generator_client, generator_model, gen_prompt, 
                "generator", call_id, log_dir
            )
            final_answer = extract_answer(gen_response)
            
            # Evaluate
            is_correct = evaluate_single_answer(final_answer, gt_answer)
            test_correct += (1 if is_correct else 0)
            test_total += 1
            test_answers.append(final_answer)
            test_targets.append(gt_answer)
            
            if (i + 1) % 20 == 0:
                current_acc = test_correct / test_total
                print(f"Test progress: {i+1}/{len(test_samples)}, Current accuracy: {current_acc:.3f}")
                
        except Exception as e:
            print(f"Error evaluating test sample {i}: {e}")
            continue
    
    # Process batched for XBRL
    processed_answers, processed_targets = process_batched(test_answers, test_targets)
    
    # Calculate final accuracy
    all_target_types = list(set(processed_targets))
    final_test_accuracy, _ = evaluate_accuracy(processed_answers, processed_targets, all_target_types)
    
    print(f"Test Evaluation Results:")
    print(f"  Raw accuracy: {test_correct}/{test_total} = {test_correct/test_total:.3f}")
    print(f"  Processed accuracy: {final_test_accuracy:.3f}")
    print(f"{'='*40}")
    
    return {
        "raw_accuracy": test_correct / test_total if test_total > 0 else 0,
        "processed_accuracy": final_test_accuracy,
        "correct_count": test_correct,
        "total_count": test_total
    }


def main():
    args = parse_args()
    generator_client, reflector_client, curator_client = initialize_clients(args.use_together_api)

    # Create timestamped run folder with key information
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    api_type = "together" if args.use_together_api else "sambanova"
    
    # Extract model name for folder (just the model part, not the full path)
    generator_short = args.generator_model.split('/')[-1] if '/' in args.generator_model else args.generator_model
    
    # Detect data type from filename
    data_type = "batched" if "batched" in args.dataset_path else "breakdown"
    
    run_folder = f"run_{timestamp}_{api_type}_gen_{generator_short}_epochs_{args.num_epochs}_samples_{args.num_samples}_{data_type}"
    
    # Create full save path
    full_save_path = os.path.join(args.save_path, run_folder)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Create detailed logs directory
    log_dir = os.path.join(full_save_path, "detailed_llm_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Results will be saved to: {full_save_path}")
    print(f"Detailed LLM logs will be saved to: {log_dir}")
    print(f"Detected data type: {data_type}")

    # Load training data
    with open(args.dataset_path, 'r') as json_file:
        all_samples = list(json_file)

    if args.num_samples != -1: 
        random.seed(42)
        random.shuffle(all_samples)
        all_samples = all_samples[:args.num_samples]

    # Load test data
    print(f"Loading test data from: {args.test_dataset_path}")
    try:
        with open(args.test_dataset_path, 'r') as json_file:
            test_samples = list(json_file)
        print(f"Loaded {len(test_samples)} test samples")
    except Exception as e:
        print(f"Warning: Could not load test data: {e}")
        test_samples = []

    # Save run configuration
    config_info = {
        "timestamp": timestamp,
        "api_type": api_type,
        "data_type": data_type,
        "generator_model": args.generator_model,
        "reflector_model": args.reflector_model,
        "curator_model": args.curator_model,
        "num_epochs": args.num_epochs,
        "num_samples": args.num_samples,
        "num_test_samples": args.num_test_samples,
        "max_num_rounds": args.max_num_rounds,
        "dataset_path": args.dataset_path,
        "test_dataset_path": args.test_dataset_path,
        "use_together_api": args.use_together_api
    }
    
    config_path = os.path.join(full_save_path, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)
    
    # Initialize tracking variables
    epoch_results = []
    current_cheatsheet = "(empty)"
    
    # Use model names directly from args
    generator_model = args.generator_model
    reflector_model = args.reflector_model
    curator_model = args.curator_model
    sleep_time = get_sleep_time(args.use_together_api)
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"STARTING EPOCH {epoch + 1}/{args.num_epochs}")
        print(f"Using models: Generator={generator_model}, Reflector={reflector_model}, Curator={curator_model}")
        print(f"Save path: {full_save_path}")
        print(f"{'='*60}")
        
        # Reset tracking for this epoch
        epoch_correct = 0
        epoch_total = 0
        epoch_answers = []
        epoch_targets = []
        question_counts = 1
        old_cheatsheet = current_cheatsheet
        
        for sample in all_samples:
            time.sleep(sleep_time) # Adaptive sleep time based on API
            print(f"==========Epoch {epoch+1}, Question {question_counts}==========")
            task_dict = ast.literal_eval(sample)
            reflection = "(empty)"
            all_context  = task_dict["context"]

            # Copy exact parsing logic from generate_cheatsheet_with_reflection.py
            # XBRL finer 
            index = all_context.index("Answer the following 4 independent questions by providing only")
            context, question = all_context[:index], all_context[index:]

            gt_answer = task_dict["target"]
       
            # get answer from generator (using generator_client)
            gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
            
            # Use timed LLM call with logging
            call_id = f"train_gen_initial_{question_counts}"
            gen_response, call_info = timed_llm_call(
                generator_client, generator_model, gen_prompt, 
                "generator", call_id, log_dir
            )
            final_answer = extract_answer(gen_response)
            
            # EVALUATE ACCURACY BEFORE REFLECTION
            is_correct = evaluate_single_answer(final_answer, gt_answer)
            epoch_correct += (1 if is_correct else 0)
            epoch_total += 1
            epoch_answers.append(final_answer)
            epoch_targets.append(gt_answer)
            
            current_accuracy = epoch_correct / epoch_total
            print(f"Current answer: {final_answer}")
            print(f"Ground truth: {gt_answer}")
            print(f"Correct: {is_correct}")
            print(f"Current epoch accuracy: {current_accuracy:.3f} ({epoch_correct}/{epoch_total})")

            if not is_correct: 
                for i in range(args.max_num_rounds):
                    # reflect (using reflector_client)
                    reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer)
                    
                    # Use timed LLM call with logging
                    call_id = f"train_reflect_{question_counts}_round_{i}"
                    reflection, call_info = timed_llm_call(
                        reflector_client, reflector_model, reflection_prompt, 
                        "reflector", call_id, log_dir
                    )
                 
                    # generate after reflection (using generator_client)
                    gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
                    
                    # Use timed LLM call with logging
                    call_id = f"train_gen_after_reflect_{question_counts}_round_{i}"
                    gen_response, call_info = timed_llm_call(
                        generator_client, generator_model, gen_prompt, 
                        "generator", call_id, log_dir
                    )
                    final_answer = extract_answer(gen_response)
                    
                    # Re-evaluate after reflection
                    if evaluate_single_answer(final_answer, gt_answer): 
                        print(f"Corrected after reflection round {i+1}!")
                        break 

            # DELTA APPROACH: Generate only the missing parts (using curator_client)
            if reflection != "(empty)":  # Only generate delta if there was a reflection
                delta_prompt = delta_curator_prompt.format(old_cheatsheet, reflection, question)
                
                # Use timed LLM call with logging
                call_id = f"train_curator_{question_counts}"
                delta_response, call_info = timed_llm_call(
                    curator_client, curator_model, delta_prompt, 
                    "curator", call_id, log_dir
                )
                delta_dict = extract_delta_json(delta_response)
                
                print(f"Delta extracted: {delta_dict}")
                
                # Merge delta into existing cheatsheet
                old_cheatsheet = merge_delta_to_cheatsheet(old_cheatsheet, delta_dict)
            
            # Save intermediate cheatsheet every 20 samples (changed from 10)
            if question_counts % 50 == 0:
                intermediate_path = os.path.join(full_save_path, f"epoch_{epoch+1}_question_{question_counts}_cheatsheet.txt")
                with open(intermediate_path, "w+") as f:
                    f.write(old_cheatsheet)
                print(f"Intermediate cheatsheet saved to {intermediate_path}")
                
            question_counts += 1

        # EPOCH TRAINING EVALUATION
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} TRAINING COMPLETED")
        print(f"{'='*60}")
        
        # Process batched for XBRL finer
        processed_answers, processed_targets = process_batched(epoch_answers, epoch_targets)
        
        # Calculate final training accuracy
        all_target_types = list(set(processed_targets))
        final_train_accuracy, _ = evaluate_accuracy(processed_answers, processed_targets, all_target_types)
        
        # EPOCH TEST EVALUATION (using generator_client)
        test_results = {}
        if test_samples:
            test_results = evaluate_test_set(generator_client, generator_model, old_cheatsheet, test_samples, args.num_test_samples, log_dir)
        
        epoch_result = {
            "epoch": epoch + 1,
            "train_accuracy": final_train_accuracy,
            "train_correct_count": epoch_correct,
            "train_total_count": epoch_total,
            "test_accuracy": test_results.get("processed_accuracy", 0),
            "test_correct_count": test_results.get("correct_count", 0),
            "test_total_count": test_results.get("total_count", 0),
            "cheatsheet_length": len(old_cheatsheet),
            "cheatsheet": old_cheatsheet
        }
        epoch_results.append(epoch_result)
        
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Training Accuracy: {final_train_accuracy:.3f} ({epoch_correct}/{epoch_total})")
        if test_samples:
            print(f"  Test Accuracy: {test_results['processed_accuracy']:.3f} ({test_results['correct_count']}/{test_results['total_count']})")
        print(f"  Cheatsheet Length: {len(old_cheatsheet)} characters")
        
        # Save epoch final cheatsheet
        epoch_cheatsheet_path = os.path.join(full_save_path, f"epoch_{epoch+1}_final_cheatsheet.txt")
        with open(epoch_cheatsheet_path, "w+") as f:
            f.write(old_cheatsheet)
        
        # Set cheatsheet for next epoch
        current_cheatsheet = old_cheatsheet
        
        # Save epoch results
        epoch_results_path = os.path.join(full_save_path, "epoch_results.json")
        with open(epoch_results_path, "w") as f:
            json.dump(epoch_results, f, indent=2)
    
    # FINAL SUMMARY
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for i, result in enumerate(epoch_results):
        print(f"Epoch {result['epoch']}:")
        print(f"  Train Accuracy: {result['train_accuracy']:.3f} ({result['train_correct_count']}/{result['train_total_count']})")
        print(f"  Test Accuracy:  {result['test_accuracy']:.3f} ({result['test_correct_count']}/{result['test_total_count']})")
        if i > 0:
            prev_train = epoch_results[i-1]['train_accuracy']
            prev_test = epoch_results[i-1]['test_accuracy']
            train_improvement = result['train_accuracy'] - prev_train
            test_improvement = result['test_accuracy'] - prev_test
            print(f"  Train Improvement: {train_improvement:+.3f}")
            print(f"  Test Improvement:  {test_improvement:+.3f}")
        print()
    
    # Save final summary
    summary_path = os.path.join(full_save_path, "training_summary.json")
    summary = {
        "run_info": config_info,
        "num_epochs": args.num_epochs,
        "num_train_samples_per_epoch": len(all_samples),
        "num_test_samples_per_epoch": args.num_test_samples,
        "epoch_results": epoch_results,
        "final_cheatsheet_length": len(current_cheatsheet)
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save final cheatsheet
    final_cheatsheet_path = os.path.join(full_save_path, "final_cheatsheet.txt")
    with open(final_cheatsheet_path, "w") as f:
        f.write(current_cheatsheet)
    
    print(f"Results saved to: {full_save_path}")
    print(f"Final cheatsheet length: {len(current_cheatsheet)} characters")


if __name__=="__main__":
    main()
