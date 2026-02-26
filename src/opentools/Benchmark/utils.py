import os
import json


class ResultAnalyzer:
    @staticmethod
    def calculate_time_steps(log_dir):
        time_list = []
        step_list = []
        files = []
        if os.path.exists(log_dir):
            files = os.listdir(log_dir)
        for file in files:
            if file.endswith(".log"):
                with open(os.path.join(log_dir, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Total steps executed" in line:
                            step_list.append(int(line.split(":")[-1].strip()))
                        if "Total execution time" in line:
                            time_list.append(float(line.split(":")[-1].strip().split(" ")[0]))

        print(f"Log dir: {log_dir}")
        # If no log-derived data, fall back to reading result JSONs
        if len(time_list) == 0 or len(step_list) == 0:
            result_dir = log_dir.replace("logs", "results")
            if os.path.exists(result_dir):
                for filename in os.listdir(result_dir):
                    if filename.endswith('.json') and filename.startswith('output_'):
                        try:
                            with open(os.path.join(result_dir, filename), 'r') as f:
                                data = json.load(f)
                            if 'total_execution_time' in data:
                                try:
                                    time_list.append(float(data['total_execution_time']))
                                except Exception:
                                    pass
                            # Prefer explicit steps, else try to infer from reasoning_trace
                            if 'steps_taken' in data:
                                try:
                                    step_list.append(int(data['steps_taken']))
                                except Exception:
                                    pass
                            elif 'reasoning_trace' in data and isinstance(data['reasoning_trace'], str):
                                inferred_steps = data['reasoning_trace'].count('Agent Step')
                                if inferred_steps > 0:
                                    step_list.append(inferred_steps)
                        except Exception:
                            continue

        # Guard against empty lists
        if len(time_list) == 0:
            average_time = 0.0
        else:
            average_time = round(sum(time_list) / len(time_list), 1)

        if len(step_list) == 0:
            average_step = 0.0
        else:
            average_step = round(sum(step_list) / len(step_list), 2)

        # count problems solved in one step
        one_step_count = sum([1 for step in step_list if step == 1])
        if len(step_list) == 0:
            one_step_rate = 0.0
        else:
            one_step_rate = round(one_step_count / len(step_list), 1)

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
                try:
                    with open(os.path.join(result_dir, filename), 'r') as f:
                        data = json.load(f)
                    total_problems += 1
                    counted = False

                    # Primary: count explicit memory tool usage
                    if 'memory' in data and isinstance(data['memory'], dict):
                        for step in data['memory'].values():
                            if isinstance(step, dict) and 'tool_name' in step:
                                tool_name = step['tool_name']
                                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                counted = True
                    try:
                        if not counted and 'reasoning_trace' in data.keys():
                            for agent_step in data['reasoning_trace'].keys():
                                steps = data['reasoning_trace'][agent_step]
                                if steps.get('Final Answer') or steps.get('final_answer'):
                                    continue
                                if "tool_calls" in steps.keys():
                                    for tool_call in steps["tool_calls"]:
                                        tool_name = tool_call["name"]
                                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                        counted = True
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error parsing tool usage: {e}")
                        if 'pid' in data:
                            print(f"ID: {data['pid']}")
                        continue
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON file {filename}: {e}. Skipping.")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing file {filename}: {e}. Skipping.")
                    continue

        # Calculate ratios
        total_tool_usage = sum(tool_usage.values())
        if total_tool_usage == 0:
            if return_counts:
                return {}, {}, 0
            return {}

        # Preserve raw counts before normalization
        counts_sorted = dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))

        for tool in tool_usage:
            tool_usage[tool] = round(tool_usage[tool] / total_tool_usage, 3)
        # Sort the dictionary by value in descending order
        sorted_tool_usage = dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))

        if return_counts:
            return sorted_tool_usage, counts_sorted, total_tool_usage
        return sorted_tool_usage