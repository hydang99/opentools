import concurrent.futures
import os
import json
import argparse
from tqdm import tqdm
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.openai import ChatOpenAI
from opentools.Benchmark.utils import ResultAnalyzer

class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

class BinaryAnswerVerification(BaseModel):
    true_false: bool


class ResultScorer:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine or ChatOpenAI(model_string="gpt-4o-mini", is_multimodal=False, enable_cache=True)
        print(f"\nLocal OpenAI engine {self.llm_engine.model_string} initialized.\n")

    def answer_verification(self,question, response, correct_answer):
        # Normalize correct_answer to a list if it's a string
        if isinstance(correct_answer, str):
            correct_answer = [correct_answer]
        
        for answer in correct_answer:
            if answer.lower() in response.lower():
                return "The answer is correct", True
        print(f"Question: {question}")
        query_prompt = f"""
        This is a MathVista answer verification task. Compare the model's response against the correct answer.
        
        Question: {question}
        Model response: {response}
        List of Correct answers: {correct_answer}

        MathVista evaluation rules:
        1. Extract the core answer from the model response (ignore explanations or additional context)
        2. The answer is correct if it matches any of the correct answers:
           - It matches the correct answer semantically 
           - It expresses the same meaning in different words 
           - It includes the correct answer with additional context that doesn't change the meaning
        3. The answer is incorrect if:
           - It conveys a different meaning
           - It's too vague or too specific compared to the correct answer
           - It contains contradictory information

        Response Format:
        <analysis>: First extract the core answer, then explain if it semantically matches any of the correct answers
        <true_false>: Return "True" if the extracted answer matches semantically, otherwise "False"
        """

        verification = self.llm_engine(query_prompt, response_format=AnswerVerification)

        analysis = verification.analysis.strip()
        true_false = verification.true_false

        return analysis, true_false

    def score_results(self, results, max_workers=10):
        correct = 0
        
        def process_single_result(pid_data):
            pid, question_data = pid_data
            question = question_data["question"]
            response = question_data["response"]
            correct_answer = question_data["correct_answer"]
            analysis, true_false = self.answer_verification(question, response, correct_answer)
            return pid, analysis, true_false
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_result, (pid, data)) 
                      for pid, data in results.items()]
            # Use tqdm with better refresh settings for gradual updates
            with tqdm(total=len(futures), desc="Scoring results", unit="result", 
                     ncols=100, miniters=1, mininterval=0.1, file=sys.stdout, 
                     dynamic_ncols=True, leave=True, smoothing=0.1) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        pid, analysis, true_false = future.result()
                        correct += 1 if true_false else 0
                        results[pid].update({
                            "stepwise_analysis": analysis,
                            "true_false": true_false
                        })
                        pbar.update(1)
                        pbar.refresh()  # Force immediate refresh to show gradual progress
                    except Exception as e:
                        print(f"\nError processing result: {e}")
                        pbar.update(1)
                        pbar.refresh()  # Force immediate refresh
        return results, correct

    @staticmethod
    def calculate_time_steps(log_dir):
        time_list = []
        step_list = []
        files = os.listdir(log_dir)
        for file in files:
            if file.endswith(".log"):
                """
                ==>Total steps executed: 4
                ==>Total execution time: 103.47 seconds
                """
                with open(os.path.join(log_dir, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Total steps executed" in line:
                            step_list.append(int(line.split(":")[-1].strip()))
                        if "Total execution time" in line:
                            time_list.append(float(line.split(":")[-1].strip().split(" ")[0]))

        print(f"Log dir: {log_dir}")
        average_time = round(sum(time_list) / len(time_list), 1)
        average_step = round(sum(step_list) / len(step_list), 2)

        # count prolems solved in one step
        one_step_count = sum([1 for step in step_list if step == 1])
        one_step_rate = round(one_step_count / len(step_list) * 100, 1) if len(step_list) > 0 else 0.0

        # save the step stats
        step_stats = {
            "average_time": average_time,
            "average_step": average_step,
            "one_step_rate": one_step_rate
        }

        return step_stats

    @staticmethod
    def calculate_tool_usage(result_dir, return_counts=False):
        """
        Calculate the usage of tools.

        If return_counts is False (default), returns a dict of ratios (normalized).
        If return_counts is True, returns a tuple of (ratios_dict, counts_dict, total_tool_usage).
        """
        tool_usage = {}
        total_problems = 0
        for filename in os.listdir(result_dir):
            if filename.endswith('.json'):
                with open(os.path.join(result_dir, filename), 'r') as f:
                    data = json.load(f)
                    total_problems += 1
                    if 'memory' in data:
                        used_tools = set()
                        for step in data['memory'].values():
                            if 'tool_name' in step:
                                tool_name = step['tool_name']
                                if tool_name not in used_tools:
                                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                    used_tools.add(tool_name)

        # Calculate ratios
        if total_problems == 0:
            if return_counts:
                return {}, {}, 0
            for tool in tool_usage:
                tool_usage[tool] = 0.0
            return dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))

        counts_sorted = dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))
        for tool in tool_usage:
            tool_usage[tool] = round(tool_usage[tool] / total_problems, 3)

        # Sort the dictionary by value in descending order
        sorted_tool_usage = dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))

        if return_counts:
            return sorted_tool_usage, counts_sorted, total_problems
        return sorted_tool_usage

def load_data(data_file, result_dir, response_type):
    # Load the benchmark data
    with open(data_file, 'r') as f:
        # convert the benchmark data to a dictionary
        benchmark_data = {data["pid"]: data for data in json.load(f)}
        # Load the results  
    results = {}
    tokens_usage = {"total_tokens": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0, "call_count": 0, "average_tokens_per_call": 0}
    tokens_retrieving_tool = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error_stats = {"error_calls": 0, "outputs_with_errors": 0, "total_outputs": 0}
    repeat_stats = {"repeat_calls": 0, "outputs_with_repeats": 0, "total_outputs": 0}
    for file in os.listdir(result_dir):
        # Only consider files that match the exact pattern: output_<number>.json
        if file.endswith(".json") and file.startswith("output_") and len(file.split("_")) == 2:
            try:
                with open(os.path.join(result_dir, file), 'r') as f:
                    result = json.load(f)
                error_execs = result.get("error_executions") or []
                repeat_in_output = False
                error_in_output = False
                if isinstance(error_execs, list):
                    for exec_err in error_execs:
                        err_msg = str(exec_err.get("error", "")).lower()
                        is_repeat = "repeat" in err_msg
                        if is_repeat:
                            repeat_stats["repeat_calls"] += 1
                            repeat_in_output = True
                        else:
                            error_stats["error_calls"] += 1
                            error_in_output = True
                if repeat_in_output:
                    repeat_stats["outputs_with_repeats"] += 1
                if error_in_output:
                    error_stats["outputs_with_errors"] += 1
                repeat_stats["total_outputs"] += 1
                error_stats["total_outputs"] += 1
                # Get the PID from the result JSON if available, otherwise use filename index
                if "pid" in result:
                    pid = str(result["pid"])
                else:
                    # For gaia-text, the output files don't have pid field, so we use the filename index
                    index = file.replace(".json", "").replace("output_", "") # "0", "1", "2", ...
                    # Make sure it's a valid number
                    if not index.isdigit(): 
                        print(f"Skipping {file} - not a valid number format")
                        continue
                    pid = str(int(index)) # NOTE adjust the index to match the pid
                tokens_usage["total_tokens"] += result.get("token_usage", {}).get("total_tokens", 0)
                tokens_usage["total_prompt_tokens"] += result.get("token_usage", {}).get("total_prompt_tokens", 0)
                tokens_usage["total_completion_tokens"] += result.get("token_usage", {}).get("total_completion_tokens", 0)
                tokens_usage["call_count"] += result.get("token_usage", {}).get("call_count", 0)

                tokens_retrieving_tool["prompt_tokens"] += result.get("tokens_retrieving_tool", {}).get("prompt_tokens", 0)
                tokens_retrieving_tool["completion_tokens"] += result.get("tokens_retrieving_tool", {}).get("completion_tokens", 0)
                tokens_retrieving_tool["total_tokens"] += result.get("tokens_retrieving_tool", {}).get("total_tokens", 0)
                # Save the results  

                if pid in benchmark_data:
                    results[pid] = benchmark_data[pid]
                    results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
                    results[pid]["question"] = benchmark_data[pid]["query"]
                    # If there's an error field (failed execution), retain it and mark placeholder response
                    if "error" in result:
                        print(f"Warning: {file} contains error: {result['error']}, counting as wrong")
                        results[pid]["load_error"] = str(result["error"])
                        results[pid]["response"] = ""
                        continue  # still counted later but not scored
                    
                    if response_type in result:
                        results[pid]["response"] = result[response_type]
                    else:
                        # Try alternative response types if the primary one is missing
                        alternative_types = ["final_answer", "direct_output", "base_response", "response","full_answer"]
                        found_alternative = False
                        
                        for alt_type in alternative_types:
                            if alt_type in result:
                                print(f"Warning: {response_type} not found in {file}, using {alt_type} instead")
                                results[pid]["response"] = result[alt_type]
                                found_alternative = True
                                break
                        
                        if not found_alternative:
                            print(f"Warning: No response field found in {file}, counting as wrong")
                            results[pid]["load_error"] = "no_response_field"
                            results[pid]["response"] = ""
                else:
                    print(f"Warning: PID {pid} not found in benchmark data")
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    tokens_usage["average_tokens_per_call"] = tokens_usage["total_tokens"] / tokens_usage["call_count"] if tokens_usage["call_count"] > 0 else 0
    return results, tokens_usage, tokens_retrieving_tool, error_stats, repeat_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and score the results from the benchmark data")
    parser.add_argument("--data_file", type=str, default="data/data.json", help="The file containing the benchmark data")
    parser.add_argument("--result_dir", type=str, default=None, help="The directory containing the results")
    parser.add_argument("--output_file", type=str, default="final_results.json", help="The file to save the extracted results")
    parser.add_argument("--log_dir", type=str, default=None, help="The directory containing the logs")
    parser.add_argument("--response_type", type=str, default="final_answer", 
                        choices=["final_answer", "direct_output", "base_response", "full_answer"],
                        help="The type of response to extract from the results")
    parser.add_argument("--max_workers", type=int, default=16, help="The maximum number of workers to use")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()

        # Load and print the arguments
        print("#"*50)
        print(f"Arguments: {args}")
        for arg, value in args.__dict__.items():
            print(f"# {arg}: {value}")
        print("#"*50)

        scorer = ResultScorer()
        analyzer = ResultAnalyzer()

        # Load the results
        try:
            results, tokens_usage, tokens_retrieving_tool, error_stats, repeat_stats = load_data(args.data_file, args.result_dir, args.response_type)
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            print("Exited.\n")
            exit()

        # Validate results before scoring
        total_loaded = len(results)
        print(f"Loaded {total_loaded} results for scoring")
        valid_results = {}
        invalid_pids = []
        for pid, data in results.items():
            if "response" in data and "correct_answer" in data:
                if data["response"] and isinstance(data["response"], str) and data["response"].strip():
                    valid_results[pid] = data
                else:
                    invalid_pids.append(pid)
                    print(f"Warning: PID {pid} has empty response, will count as wrong")
            else:
                invalid_pids.append(pid)
                print(f"Warning: PID {pid} missing required fields, will count as wrong")

        print(f"Valid results for scoring: {len(valid_results)}")

        # Score only valid results
        scored_results, correct = scorer.score_results(valid_results, max_workers=args.max_workers) if len(valid_results) > 0 else ({}, 0)

        # Merge scored results back into full set and mark invalids as wrong
        for pid, data in scored_results.items():
            results[pid] = data
        for pid in invalid_pids:
            # Ensure fields exist for saving and downstream analysis
            if pid not in results:
                results[pid] = {"pid": pid}
            reason = results[pid].get("load_error", "invalid_or_missing_response")
            results[pid]["stepwise_analysis"] = f"Not scored: {reason}"
            results[pid]["true_false"] = False

        # Calculate accuracy and wrong answers
        denominator = total_loaded if total_loaded > 0 else 1
        acc = round(correct / denominator * 100, 2)
        print(f"\nAccuracy: {acc}% ({correct}/{denominator})")

        # Save detailed results
        output_file = os.path.join(args.result_dir, args.output_file)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
            print(f"\nResults saved to {output_file}")

        # Calculate wrong answers
        wrong_pids = []
        for pid, data in results.items():
            if "true_false" not in data:
                print(f"Warning: PID {pid} missing 'true_false' field")
                continue
            if not data["true_false"]:
                wrong_pids.append(pid)
        
        wrong_pids = sorted(wrong_pids, key=lambda x: int(x))
        wrong_indices = [int(pid) for pid in wrong_pids]
        print(f"Wrong PIDs: {wrong_pids}")
        print(f"Wrong Indices: {wrong_indices}")

        if error_stats.get("total_outputs", 0) > 0:
            error_stats["error_output_rate"] = round(
                error_stats["outputs_with_errors"] / error_stats["total_outputs"], 4
            )
            error_stats["error_call_rate_per_output"] = round(
                error_stats["error_calls"] / error_stats["total_outputs"], 4
            )
        else:
            error_stats["error_output_rate"] = 0.0
            error_stats["error_call_rate_per_output"] = 0.0

        if repeat_stats.get("total_outputs", 0) > 0:
            repeat_stats["repeat_output_rate"] = round(
                repeat_stats["outputs_with_repeats"] / repeat_stats["total_outputs"], 4
            )
            repeat_stats["repeat_call_rate_per_output"] = round(
                repeat_stats["repeat_calls"] / repeat_stats["total_outputs"], 4
            )
        else:
            repeat_stats["repeat_output_rate"] = 0.0
            repeat_stats["repeat_call_rate_per_output"] = 0.0

        scores = {
            "correct": correct,
            "total": len(results),
            "accuracy": acc,
            "wrong_pids": wrong_pids,
            "wrong_indices": wrong_indices,
            "tokens_usage": tokens_usage,
            "tokens_retrieving_tool": tokens_retrieving_tool,
            "error_executions": error_stats,
            "repeat_tool_errors": repeat_stats
        }

        # Calculate additional statistics if log directory is provided
        log_dir = args.log_dir or args.result_dir.replace("results", "logs")
        if os.path.exists(log_dir):

            if args.response_type == "base_response":
                print("Base response is not supported for scoring.")
                print("Exited.\n")
                exit()

            # Calculate the average time and steps (robust to empty logs)
            try:
                step_stats = analyzer.calculate_time_steps(log_dir)
            except Exception as _calc_err:
                print(f"Warning: failed to compute step stats: {_calc_err}")
                step_stats = {"average_time": 0.0, "average_step": 0.0, "one_step_rate": 0.0}
            print(f"\nStep stats:")
            for key, value in step_stats.items():
                print(f"- {key}: \t{value}")

            # Calculate the usage of tools (robust to zero totals)
            try:
                tool_usage, tool_usage_counts, tool_usage_total = analyzer.calculate_tool_usage(args.result_dir, return_counts=True)
            except Exception as _tool_err:
                print(f"Warning: failed to compute tool usage: {_tool_err}")
                tool_usage, tool_usage_counts, tool_usage_total = {}, {}, 0
            print(f"\nTool usage:")
            for tool, ratio in tool_usage.items():
                print(f"- {tool}: \t{ratio}")

            # Update the scores 
            scores.update({
                "step_stats": step_stats,
                "tool_usage": tool_usage,
                "tool_usage_counts": tool_usage_counts,
                "tool_usage_total_steps": tool_usage_total
            })
        print("\nTokens usage:")
        for key, value in tokens_usage.items():
            print(f"- {key}: \t{value}")
        print("\nError executions:")
        for key, value in error_stats.items():
            print(f"- {key}: \t{value}")
        print("\nRepeat tool errors:")
        for key, value in repeat_stats.items():
            print(f"- {key}: \t{value}")
        print("\nError executions:")
        for key, value in error_stats.items():
            print(f"- {key}: \t{value}")
        # Save the scores
        try:
            score_file = os.path.join(args.result_dir, f"final_scores_{args.response_type}.json")
            with open(score_file, 'w') as f:
                json.dump(scores, f, indent=4)
                print(f"Scores saved to {score_file}")
        except Exception as e:
            print(f"Error saving scores: {str(e)}")
            print("Exited.\n")
            exit()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print("Exited.\n")
        exit()
