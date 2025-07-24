from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
import argparse
import re 
import time 
import torch 
from tqdm import tqdm
import sklearn
import evaluate
import openai 
from glob import glob
import ast
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset, load_from_disk
from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import eval_for_GameOf24, eval_for_multiple_choice, eval_for_exact_matching_with_no_punctuation, eval_equation_balancer

from dotenv import load_dotenv
from utils_claude_revised import * 



def save_detailed_log(detailed_log_dir, question_idx, question_data):
    """
    Save detailed log for each question including prompt, response, and other details
    """
    if detailed_log_dir is None:
        return
    
    log_filename = f"question_{question_idx:04d}.json"
    log_filepath = os.path.join(detailed_log_dir, log_filename)
    
    # Add timestamp to the log
    question_data["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    question_data["question_index"] = question_idx
    
    # Save to JSON file with pretty formatting
    with open(log_filepath, "w", encoding="utf-8") as f:
        json.dump(question_data, f, indent=2, ensure_ascii=False)

#### API key information ####

api_key = os.environ['SAMBANOVA_API_KEY']
base_url = "https://api.sambanova.ai/v1"

###---------------------####

DATA_DIR = "./data/finlora/test/" 

# Map task names to their JSONL files in the data/test directory
dataset_path = {
    "xbrl_tags_extract":         os.path.join(DATA_DIR, "xbrl_extract_tags_test.jsonl"),
    "xbrl_value_extract":        os.path.join(DATA_DIR, "xbrl_extract_value_test.jsonl"),
    "xbrl_formula_extract":      os.path.join(DATA_DIR, "xbrl_extract_formula_test.jsonl"),
    "xbrl_formula_calc_extract": os.path.join(DATA_DIR, "xbrl_extract_formula_calculations_test.jsonl"),
    "xbrl_finer":                os.path.join(DATA_DIR, "finer_test_batched.jsonl"),
    "xbrl_fnxl":                 os.path.join(DATA_DIR, "fnxl_test_batched.jsonl"),
    "fpb":                       os.path.join(DATA_DIR, "fpb_test.jsonl"),
    "fiqa":                      os.path.join(DATA_DIR, "fiqa_test.jsonl"),
    "tfns":                      os.path.join(DATA_DIR, "tfns_test.jsonl"),
    "nwgi":                      os.path.join(DATA_DIR, "nwgi_test.jsonl"),
    "headline":                  os.path.join(DATA_DIR, "headline_test.jsonl"),
    "ner":                       os.path.join(DATA_DIR, "ner_test.jsonl"),
    "financebench":              os.path.join(DATA_DIR, "financebench_test.jsonl"),
    "xbrl_term":                 os.path.join(DATA_DIR, "xbrl_term_test.jsonl"),
    "formula":                   os.path.join(DATA_DIR, "formula_test.jsonl"),
    "cfa_level1":                os.path.join(DATA_DIR, "cfa_level1_test.jsonl"),
    "cfa_level2":                os.path.join(DATA_DIR, "cfa_level2_test.jsonl"),
    "cfa_level3":                os.path.join(DATA_DIR, "cfa_level3_test.jsonl"),
    "cpa_reg":                   os.path.join(DATA_DIR, "cpa_reg_test.jsonl"),
}


max_new_token_dict = {
    "xbrl_tags_extract": 20,
    "xbrl_value_extract": 20,
    "xbrl_formula_extract": 30,
    "xbrl_formula_calc_extract": 30,
    "xbrl_finer": 100,
    "xbrl_fnxl": 100,
    "fpb": 10,
    "fiqa": 10,
    "tfns": 10,
    "nwgi": 10,
    "headline": 10,
    "ner": 10,
    "financebench": 50,
    "xbrl_term": 50,
    "formula": 50
}

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Arguments to pass to the program.")

    # Dataset name
    parser.add_argument("--dataset", required=True, help="Comma-separated list of dataset keys")

    # Approach name
    parser.add_argument("--approach_name", type=str, default="DynamicCheatsheet_Cumulative", help="Approach name")

    # Model name
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o-mini", help="Model name")

    # Paths to the prompt files
    parser.add_argument("--generator_prompt_path", type=str, default="prompts/simple_generator.txt", help="Path to the generator prompt file")
    parser.add_argument("--cheatsheet_prompt_path", type=str, default=None, help="Path to the cheatsheet prompt file")
    
    # ADD THIS NEW PARAMETER FOR FIXED CHEATSHEET
    parser.add_argument("--fixed_cheatsheet_path", type=str, default=None, help="Path to the fixed cheatsheet file (for fixed_cheatsheet approach)")
    
    # Detailed logging
    parser.add_argument("--detailed_log", action="store_true", help="Enable detailed logging of prompts and responses")

    # Additional model-related arguments
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max_num_rounds", type=int, default=1, help="Maximum number of rounds")

    # Execution flags
    parser.add_argument("--execute_python_code", action="store_true", help="Allow Python code execution")
    parser.add_argument("--initialize_cheatsheet_path", type=str, default=None, help="Path to initialize the cheatsheet")
    parser.add_argument("--retrieve_top_k", type=int, default=3, help="Top-k retrieval for dynamic approaches")

    # Continue from the previous run
    parser.add_argument("--continue_from_last_run_path", type=str, default=None, help="Path to continue from the last run")

    # Additional save-path-related arguments
    parser.add_argument("--save_directory", type=str, default="results", help="Directory to save results")
    parser.add_argument("--additional_flag_for_save_path", type=str, default="", help="Additional flag for the save path")
    
    # Additional inference-time args 
    parser.add_argument("--max_n_samples", type=int, default=-1, help="Maximum number of samples to process")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_sample", type=int, default=500)
    parser.add_argument("--top_k_aggregate", type=int, default=5)
    args = parser.parse_args()

    # Convert to a dictionary for compatibility with the rest of the code
    return args

def read_file(file_path: str) -> str:
    """
    Read the file and return the content.
    """
    with open(file_path, "r") as file:
        return file.read()

    
def write_jsonl(file_path, data):
    """
    Save the outputs to a file.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def initialize_client():
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client


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


def evaluate_accuracy(out, target, target_type_list):
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
    assert len(processed_out_text_list) == len(processed_target_list)
    return processed_out_text_list, processed_target_list


def api_with_backoff(client, model, prompt):
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        # Calculate exponential backoff (base 2)
        sleep_time = min(2 ** retry_count, 60)  # Cap at 60 seconds to avoid too long waits
        print(f"Attempt {retry_count + 1} failed. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        retry_count += 1
    return response

def embedding_with_backoff(client, prompt):
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.embeddings.create(
                model="E5-Mistral-7B-Instruct",
                input=[prompt]
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        # Calculate exponential backoff (base 2)
        sleep_time = min(2 ** retry_count, 60)  # Cap at 60 seconds to avoid too long waits
        print(f"Attempt {retry_count + 1} failed. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        retry_count += 1
    return response.data[0].embedding


def test_fin_tasks(args, data_name="xbrl_finer", prompt_fun=None):
    start_time = time.time()
    results = {}
    
    print(f"Testing model: {args.model_name} on {data_name} with temperature={args.temperature}")

    if data_name not in dataset_path.keys():
        return results

    instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
    sample_size = len(instructions)

    if args.sample_ratio < 1.0:
        sample_size = int(len(instructions) * args.sample_ratio)
        instructions = instructions.sample(frac=args.sample_ratio, random_state=42)

    # Read the prompt files
    # args.generator_prompt = read_file(args.generator_prompt_path)
    # if args.cheatsheet_prompt_path:
    #     args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
    # else:
    #     args.cheatsheet_prompt = "(empty)"

    # Initialize the language model
    if args.approach_name in ["test_with_global_cheatsheet", "no_cheatsheet", "fixed_cheatsheet"]:
        model = initialize_client()
    else:
        model = LanguageModel(
            model_name=args.model_name,
        )

    # Initialize the cheatsheet
    cheatsheet = "(empty)"
    reflection = "(empty)"

    all_cheatsheet = []
    if args.cheatsheet_prompt_path is not None:
        if os.path.isfile(args.cheatsheet_prompt_path):
            with open(args.cheatsheet_prompt_path, "r") as file:
                cheatsheet = file.read()
        else:
            print("Getting all the delta cheatsheets...")
            files = glob(args.cheatsheet_prompt_path + "/*")
            for file_path in files:
                with open(file_path, 'r') as f:
                    all_cheatsheet.append(f.read())
    print(f"Length of all cheatsheets: {len(all_cheatsheet)}")
    print("Getting all question embeddings...")
    embeddings_500 = []
    with open("data/finlora/train/finer_train_batched_embeddings_500.txt", 'r') as f:
        for line in f:
            embeddings_500.append(ast.literal_eval(line))
    embeddings_500 = embeddings_500[:args.num_train_sample]
    print(f"Length of all question embeddings: {len(embeddings_500)}")

    time_stamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
    args.save_path_name = f"{args.save_directory}/{data_name}/{args.model_name}_{args.approach_name}_{time_stamp}.jsonl"

    # Create the directory if it does not exist
    dir_path = os.path.dirname(args.save_path_name)
    os.makedirs(dir_path, exist_ok=True)

    save_param_path = args.save_path_name.replace(".jsonl", "_params.json")
    dir_path = os.path.dirname(save_param_path)
    os.makedirs(dir_path, exist_ok=True)

     # Save the arguments to a file
    with open(save_param_path, "w") as file:
        json.dump(vars(args), file, indent=4)

    # Create detailed log directory if detailed_log is enabled
    detailed_log_dir = None
    if args.detailed_log:
        detailed_log_dir = f"{args.save_directory}/{data_name}/detailed_logs/{args.model_name}_{args.approach_name}_{time_stamp}"
        os.makedirs(detailed_log_dir, exist_ok=True)
        print(f"Detailed logs will be saved to: {detailed_log_dir}")

    # Initialize the questions and the embeddings
    questions = None
    embeddings = None

    '''
    if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"]:
        df = pd.read_csv(f"embeddings/{args.task}.csv")
        questions = df["input"].tolist()
        embeddings = df["embedding"]
        embeddings = embeddings.apply(eval)
        embeddings = np.array(embeddings.tolist()) # (N, 1536)

        # Re-order the embeddings based on the order of the dataset inputs
        dataset_inputs = [example["input"] for example in dataset]
        indices = [questions.index(input) for input in dataset_inputs]
        embeddings = embeddings[indices]
        questions = dataset_inputs
    else:
        questions = [example["input"] for example in dataset]
    '''

    task_start_time = time.time()

    context = instructions['context'].tolist()
    target_list = instructions["target"].tolist()
    target_list = [str(x) for x in target_list]

    total_steps = instructions.shape[0] 
    out_text_list = []
    
    outputs = []
    generator_outputs_so_far = []

    task_pbar = tqdm(range(total_steps))
    #cheatsheet = "(empty)"
    count = 0 
    for i in task_pbar:
        tmp_context = context[i]

        if not tmp_context:
            break
        
        tmp_target = instructions['target'].tolist()[i]
        
        # time.sleep(10)
    
        if args.approach_name == "test_with_global_cheatsheet":
            # XBRL finer 
            index = tmp_context.index("Answer the following 4 independent questions by providing only")
            question_context, question = tmp_context[:index], tmp_context[index:]

            if len(all_cheatsheet) > 0:
                close_cheatsheet = []
                question_embedding = embedding_with_backoff(model, question)
                similarity = cosine_similarity(np.array([question_embedding]), np.array(embeddings_500))
                close_questions_ranked = np.argsort(similarity[0])[::-1]//args.batch_size
                close_questions = set()
                for index in close_questions_ranked:
                    if index not in close_questions:
                        # edge case for delta cheatsheet num train sample not divisible by batch size, dropping last cheatsheet
                        if index >= len(all_cheatsheet):
                            continue
                        close_cheatsheet.append(all_cheatsheet[index])
                        close_questions.add(index)
                    if len(close_cheatsheet) == args.top_k_aggregate:
                        break
                agg_prompt = aggregator_prompt.format(close_cheatsheet[0], close_cheatsheet[1], close_cheatsheet[2])
                response = api_with_backoff(model, "Llama-4-Maverick-17B-128E-Instruct", agg_prompt)
                cheatsheet = response.choices[0].message.content

            gen_prompt = generator_prompt.format(cheatsheet, reflection, question, question_context)
            response = api_with_backoff(model, args.model_name, gen_prompt)
            try:
                gen_response = response.choices[0].message.content
                count += 1
            except: 
                continue 
        
            final_answer = extract_answer(gen_response)
            output_dict = {}
            output_dict["final_output"] = gen_response 
            output_dict["final_answer"] = final_answer
            output_dict["final_cheatsheet"] = cheatsheet
            
            # Save detailed log if enabled
            if args.detailed_log:
                detailed_log_data = {
                    "approach_name": args.approach_name,
                    "model_name": args.model_name,
                    "input_context": tmp_context,
                    "question": question,
                    "question_context": question_context,
                    "target_answer": tmp_target,
                    "cheatsheet_used": cheatsheet,
                    "reflection_used": reflection,
                    "generator_prompt": gen_prompt,
                    "model_response": gen_response,
                    "final_answer": final_answer,
                    "is_correct": str(final_answer).lower().strip() == str(tmp_target).lower().strip(),
                    "aggregator_prompt": agg_prompt if len(all_cheatsheet) > 0 else None,
                    "selected_cheatsheets": close_cheatsheet if len(all_cheatsheet) > 0 else None
                }
                save_detailed_log(detailed_log_dir, i, detailed_log_data)
            
        elif args.approach_name == "no_cheatsheet":
            # No cheatsheet approach - use empty cheatsheet
            index = tmp_context.index("Answer the following 4 independent questions by providing only")
            question_context, question = tmp_context[:index], tmp_context[index:]
            
            # Set cheatsheet to empty
            cheatsheet = "(empty)"
            
            gen_prompt = generator_prompt.format(cheatsheet, reflection, question, question_context)
            response = api_with_backoff(model, args.model_name, gen_prompt)
            try:
                gen_response = response.choices[0].message.content
                count += 1
            except: 
                continue 
        
            final_answer = extract_answer(gen_response)
            output_dict = {}
            output_dict["final_output"] = gen_response 
            output_dict["final_answer"] = final_answer
            output_dict["final_cheatsheet"] = cheatsheet
            
            # Save detailed log if enabled
            if args.detailed_log:
                detailed_log_data = {
                    "approach_name": args.approach_name,
                    "model_name": args.model_name,
                    "input_context": tmp_context,
                    "question": question,
                    "question_context": question_context,
                    "target_answer": tmp_target,
                    "cheatsheet_used": cheatsheet,
                    "reflection_used": reflection,
                    "generator_prompt": gen_prompt,
                    "model_response": gen_response,
                    "final_answer": final_answer,
                    "is_correct": str(final_answer).lower().strip() == str(tmp_target).lower().strip(),
                }
                save_detailed_log(detailed_log_dir, i, detailed_log_data)
            
        elif args.approach_name == "fixed_cheatsheet":
            # Fixed cheatsheet approach - use the provided fixed cheatsheet
            index = tmp_context.index("Answer the following 4 independent questions by providing only")
            question_context, question = tmp_context[:index], tmp_context[index:]
            
            # Load fixed cheatsheet from file
            if args.fixed_cheatsheet_path and os.path.isfile(args.fixed_cheatsheet_path):
                with open(args.fixed_cheatsheet_path, "r") as file:
                    cheatsheet = file.read()
            else:
                print(f"Warning: Fixed cheatsheet path not provided or file not found: {args.fixed_cheatsheet_path}")
                cheatsheet = "(empty)"
            
            gen_prompt = generator_prompt.format(cheatsheet, reflection, question, question_context)
            response = api_with_backoff(model, args.model_name, gen_prompt)
            try:
                gen_response = response.choices[0].message.content
                count += 1
            except: 
                continue 
        
            final_answer = extract_answer(gen_response)
            output_dict = {}
            output_dict["final_output"] = gen_response 
            output_dict["final_answer"] = final_answer
            output_dict["final_cheatsheet"] = cheatsheet
            
            # Save detailed log if enabled
            if args.detailed_log:
                detailed_log_data = {
                    "approach_name": args.approach_name,
                    "model_name": args.model_name,
                    "input_context": tmp_context,
                    "question": question,
                    "question_context": question_context,
                    "target_answer": tmp_target,
                    "cheatsheet_used": cheatsheet,
                    "reflection_used": reflection,
                    "generator_prompt": gen_prompt,
                    "model_response": gen_response,
                    "final_answer": final_answer,
                    "is_correct": str(final_answer).lower().strip() == str(tmp_target).lower().strip(),
                    "fixed_cheatsheet_path": args.fixed_cheatsheet_path
                }
                save_detailed_log(detailed_log_dir, i, detailed_log_data)
            
        else:
            output_dict = model.advanced_generate(
                approach_name=args.approach_name,
                input_txt=tmp_context,
                cheatsheet=cheatsheet,
                generator_template=args.generator_prompt,
                cheatsheet_template=args.cheatsheet_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_num_rounds=args.max_num_rounds,
                allow_code_execution=args.execute_python_code,
                code_execution_flag="EXECUTE CODE!",
                #original_input_corpus=questions[:i+1],
                #original_input_embeddings=embeddings[:i+1] if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"] else None,
                generator_outputs_so_far=generator_outputs_so_far,
                retrieve_top_k=args.retrieve_top_k,
            )

        generator_outputs_so_far.append(output_dict["final_output"])

        outputs.append({
                "input": tmp_context,
                "target": tmp_target,
                **output_dict,
            })

        cheatsheet = output_dict["final_cheatsheet"]
        final_answer = output_dict["final_answer"]

        # print(f"@ CHEATSHEET:\n{cheatsheet}\n")
        # print('- ' * 50)
        # print(f"INPUT: {tmp_context}")
        # print(f"TARGET: {tmp_target}")
        # print(f"FINAL ANSWER: {final_answer}")
        # print("**" * 50)
        
        with open(args.save_path_name, "a") as f:
            f.write(f"INDEX: {i}\n")
            f.write(f"@ CHEATSHEET:\n{cheatsheet}\n")
            f.write('- ' * 50)
            f.write("\n")
            f.write(f"INPUT: {tmp_context}\n")
            f.write(f"TARGET: {tmp_target}\n")
            f.write(f"FINAL ANSWER: {final_answer}\n")
            f.write("**" * 50)
            f.write("\n")

        out_text_list.append(final_answer)

    if "finer" in data_name or "fnxl" in data_name:
        out_text_list, target_list = process_batched(out_text_list, target_list)

    per_question_time = (time.time() - task_start_time) / sample_size

    final_results = {}

    if data_name == "financebench" or data_name == "xbrl_term":
        metric = evaluate.load("bertscore")
        results = metric.compute(predictions=out_text_list, references=target_list, model_type="ProsusAI/finbert")
        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])
        final_results['precision'] = precision 
        final_results['recall'] = recall 
        final_results['f1'] = f1
        print(
            f"\n✓ {data_name}: precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, Time per question: {per_question_time:.2f}")

    else:
        all_target_type_for_classification = list(set(target_list))
        acc, response = evaluate_accuracy(out_text_list, target_list, all_target_type_for_classification)

        try:
            f1 = sklearn.metrics.f1_score(target_list, response, average='weighted')
        except:
            f1 = -1
            print(f"Error calculating F1 score for {data_name}")
        print(
                f"\n✓ {data_name}: Accuracy: {acc * 100:.3f}%, F1: {f1:.3f}, Time per question: {per_question_time:.2f} s, total count: {count}")
        
        final_results['acc'] = acc
        final_results['f1'] = f1
        results = {"task": data_name, "acc": acc, "f1": f1, "time": per_question_time}

    with open(args.save_path_name, "a") as f:
        f.write(f"Task: {data_name}\n")
        for key in final_results:
            f.write(f"{key}: {final_results[key]}\n")
        f.write(f"Per question time: {per_question_time:.2f} minutes\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Sample Ratio: {args.sample_ratio}\n")
        f.write(f"Temperature: {args.temperature}\n")

    return results


def main(args):
    with torch.no_grad():
        for dataset in args.dataset.split(','):
            print("testing:", dataset)
            test_fin_tasks(args, data_name=dataset)
    print("Evaluation Ends.")



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
