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
from collections import defaultdict
import matplotlib.pyplot as plt

class HallusionVDAnswerVerification(BaseModel):
    final_answer: bool

class ResultScorer:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine or ChatOpenAI(model_string="gpt-5-mini", is_multimodal=False, use_cache=False)
        print(f"\nLocal OpenAI engine {self.llm_engine.model_string} initialized.\n")

    def answer_verification(self, question, response, correct_answer):
        query_prompt = f"""
        Extract the final True/False answer from this response.
        Question: {question}
        Model response: {response}
        Return only the True/False answer from the response, ignoring all explanations.
        """

        response = self.llm_engine.generate(query_prompt, response_format=HallusionVDAnswerVerification)
        try:
            prediction = response.final_answer
            is_correct = prediction == bool(correct_answer)
            return prediction, is_correct
        except:
            print(f"Error: {response}")
            return None, False


    def score_results(self, results, max_workers=10):
        correct = 0
        
        def process_single_result(pid_data):
            pid, question_data = pid_data
            query = question_data["query"]
            response = question_data["response"]
            correct_answer = question_data["correct_answer"]
            analysis, true_false = self.answer_verification(query, response, correct_answer)
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

    

def load_data(data_file, result_dir, response_type):
    # Load the benchmark data
    with open(data_file, 'r') as f:
        # convert the benchmark data to a dictionary with string keys
        benchmark_data = {str(data["pid"]): data for data in json.load(f)}
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
                # Get the index of the result - only for files like output_0.json, output_1.json, etc.
                index = file.replace(".json", "").replace("output_", "") # "0", "1", "2", ...
                # Make sure it's a valid number
                if not index.isdigit(): 
                    print(f"Skipping {file} - not a valid number format")
                    continue
                    
                pid = str(int(index)) # NOTE adjust the index to match the pid
                
                # Check if the result has a pid field, otherwise use the filename index
                if "pid" in result:
                    assert result["pid"] == benchmark_data[pid]["pid"]
                else:
                    # For gaia-text, the output files don't have pid field, so we use the filename index
                    pass
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
                    
                    # If there's an error field (failed execution), retain it and mark placeholder response
                    if "error" in result:
                        print(f"Warning: {file} contains error: {result['error']}, counting as wrong")
                        results[pid]["load_error"] = str(result["error"])
                        results[pid]["response"] = ""
                        results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
                        continue  # still counted later but not scored
                    
                    if response_type in result:
                        results[pid]["response"] = result[response_type]
                        results[pid]["correct_answer"] = benchmark_data[pid]["answer"]           
                    else:
                        # Try alternative response types if the primary one is missing
                        alternative_types = ["final_answer", "direct_output", "base_response", "response","full_answer"]
                        found_alternative = False
                        
                        for alt_type in alternative_types:
                            if alt_type in result:
                                print(f"Warning: {response_type} not found in {file}, using {alt_type} instead")
                                results[pid]["response"] = result[alt_type]
                                results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
                                found_alternative = True
                                break
                        
                        if not found_alternative:
                            print(f"Warning: No response field found in {file}, counting as wrong")
                            results[pid]["load_error"] = "no_response_field"
                            results[pid]["response"] = ""
                            results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
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
        acc = (correct / len(results) * 100) if len(results) > 0 else 0.0
        print(f"\nAccuracy: {acc}% ({correct}/{len(results)})")
        
        # Calculate accuracy by category
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for pid, data in results.items():
            category = data.get("category", "unknown")
            is_correct = data.get("true_false", False)
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1
        
        # Calculate accuracy per category
        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                category_accuracies[category] = round(stats["correct"] / stats["total"] * 100, 2)
            else:
                category_accuracies[category] = 0.0
        
        # Print category accuracies
        print(f"\nAccuracy by category:")
        for category in sorted(category_accuracies.keys()):
            stats = category_stats[category]
            print(f"- {category}: {category_accuracies[category]}% ({stats['correct']}/{stats['total']})")
        
        # Plot bar chart for category accuracies
        if category_accuracies:
            try:
                categories = sorted(category_accuracies.keys())
                accuracies = [category_accuracies[cat] for cat in categories]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(categories, accuracies, color='steelblue', edgecolor='black', alpha=0.7)
                
                # Add value labels on top of bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{acc}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                plt.xlabel('Category', fontsize=12, fontweight='bold')
                plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
                plt.title('Accuracy by Category', fontsize=14, fontweight='bold')
                plt.ylim(0, max(accuracies) * 1.15 if accuracies else 100)
                plt.grid(axis='y', alpha=0.3, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save the chart
                chart_file = os.path.join(args.result_dir, f"accuracy_by_category_{args.response_type}.png")
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                print(f"\nCategory accuracy chart saved to {chart_file}")
                plt.close()
            except Exception as e:
                print(f"Warning: Failed to create category accuracy chart: {str(e)}")

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
            "accuracy": (correct / len(results) * 100) if len(results) > 0 else 0.0,
            "wrong_pids": wrong_pids,
            "wrong_indices": wrong_indices,
            "tokens_usage": tokens_usage,
            "tokens_retrieving_tool": tokens_retrieving_tool,
            "category_accuracies": category_accuracies,
            "category_stats": {cat: dict(stats) for cat, stats in category_stats.items()},
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
