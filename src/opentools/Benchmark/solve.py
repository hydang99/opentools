import os, sys, json, argparse, time, traceback
from typing import List, Dict, Any
# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from opentools.solver import UnifiedSolver
from opentools.agents import list_agents


class MultiAgentBenchmarkSolver:
    """
    Multi-agent benchmark solver that can evaluate different agents on the same benchmark dataset.
    Provides comparison capabilities and unified output formatting.
    """
    
    def __init__(
        self,
        task: str,
        agents: List[str] = ["opentools", "octotools", "react", "chain_of_thought", "zero_shot"],
        llm_engine_name: str = "gpt-4o-mini",
        data_file: str = "data/data.json",
        task_description: str = "",
        output_types: str = "base,final,direct,full",
        index: int = 0,
        verbose: bool = False,
        max_steps: int = 10,
        max_time: int = 60,
        max_tokens: int = 4000,
        output_json_dir: str = "results",
        root_cache_dir: str = "cache",
        enabled_tools: str = "",
        error_json_dir: str = "error_json",
        enable_faiss_retrieval: bool = False
    ):  
        try:
            self.agents = agents
            self.llm_engine_name = llm_engine_name
            self.task = task
            self.data_file = data_file
            self.task_description = task_description
            self.index = index
            self.verbose = verbose
            self.max_steps = max_steps
            self.max_time = max_time
            self.max_tokens = max_tokens
            self.output_json_dir = output_json_dir
            self.error_json_dir = error_json_dir
            self.root_cache_dir = root_cache_dir
            self.enabled_tools = enabled_tools.split(",") if enabled_tools else []      
            self.enable_faiss_retrieval = enable_faiss_retrieval
            self.output_types = output_types.lower().split(',')
            assert all(output_type in ["base", "final", "direct", "full"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct', 'full'."
            # Load the benchmark data and initialize the solvers
            self.benchmark_data = self.load_benchmark_data()
            self.solvers = self._initialize_solvers()
            
            if self.verbose:
                print(f"MultiAgentBenchmarkSolver initialized with {len(self.agents)} agents: {self.agents}")
                print(f"Available agents: {[agent['name'] for agent in list_agents()]}")
        except Exception as e:
            print(f"Error initializing MultiAgentBenchmarkSolver: {e}")
            raise e

    def _initialize_solvers(self) -> Dict[str, UnifiedSolver]:
        """Initialize UnifiedSolver instances for each agent"""
        try:
            solvers = {}
            
            for agent_name in self.agents:
                try:
                    # Create solver with agent-specific configuration
                    solver_kwargs = {
                        "llm_engine_name": self.llm_engine_name,
                        "verbose": self.verbose,
                        "max_steps": self.max_steps,
                        "max_time": self.max_time,
                        "root_cache_dir": self.root_cache_dir
                    }
                    
                    # Add tool configuration for tool-based agents
                    if agent_name in ["opentools", "octotools", "react"]:
                        solver_kwargs["enabled_tools"] = self.enabled_tools
                        solver_kwargs["enable_faiss_retrieval"] = self.enable_faiss_retrieval
                    
                    solvers[agent_name] = UnifiedSolver(
                        agent_name=agent_name,
                        **solver_kwargs
                    )
                    
                    if self.verbose:
                        print(f"✅ Initialized solver for agent: {agent_name}")
                        
                except Exception as e:
                    print(f"❌ Failed to initialize solver for agent '{agent_name}': {e}")
                    continue
            
            return solvers
        except Exception as e:
            print(f"Error initializing solvers: {e}")
            raise e

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        """Load benchmark data with task description"""
        # Add task description to the query
        try:
            if self.task_description:
                print(f"Task description: {self.task_description}")
                self.task_description = f"Task description: {self.task_description}\n"

            with open(self.data_file, 'r') as f:
                data = json.load(f) 
            for problem in data:
                problem['query'] = problem['query'] if 'query' in problem else problem['question']
                if self.task_description:
                    problem['query'] = self.task_description + problem['query']
                if 'unit' in problem and problem['unit'] not in [None, ""]:
                    problem['query'] = problem['query'] + f". Please answer in the unit of {problem['unit']}"
                if 'image' in problem and problem['image'] not in [None, ""]:
                    # NOTE: This is a hack to make the absolute image path relative to ưthe data file
                    problem['image'] = os.path.abspath(os.path.join(os.path.dirname(self.data_file), problem['image']))
                    assert os.path.exists(problem['image']), f"Error: Image file {problem['image']} does not exist."
            return data
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
            raise e

    def solve(self):
        """Solve benchmark problems with multiple agents"""
        try:
            total_problems = len(self.benchmark_data)

            # Solve a single problem
            if self.index is not None:
                if not 0 <= self.index < total_problems:
                    print(f"Error: Invalid problem index {self.index}. Valid indices are 0 to {total_problems-1}).")
                else:
                    self.solve_single_problem(self.index)
                return
        except Exception as e:
            print(f"Error solving benchmark problems: {e}")
            raise e

    def solve_single_problem(self, index: int):
        """
        Solve a single problem with all agents and compare results.
        
        Args:
            index (int): Index of the problem to solve
        """
        # Get the problem
        problem = self.benchmark_data[index]
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem.get('image', None)
        pid = problem.get('pid', None)
        answer = problem.get('answer', None)

        if self.verbose:
            print("\n\n")
            print("#"*100)
            print(f"## Problem {index}:")
            print(f"Question:\n{question}")
            print(f"Image: {image_path}")
            print("#"*100)

        # Initialize results structure
        
        results = {
            "pid": pid,
            "query": question,
            "image": image_path,
            "expected_answer": answer,
            f"{self.output_types[0]}_output": {},
            "metadata": {
                "task": self.task,
                "agents_used": list(self.solvers.keys()),
                "llm_engine": self.llm_engine_name,
                "max_steps": self.max_steps,
                "max_time": self.max_time,
                "enabled_tools": self.enabled_tools
            }
        }

        if 'metadata' in problem:
            results['problem_metadata'] = problem['metadata']

        # Solve with each agent
        agent_results = {}
        for agent_name, solver in self.solvers.items():
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Solving with agent: {agent_name}")
                print(f"{'='*50}")
            result = None
            try:
                # Solve with current agent
                error_executions = None
                agent_result = solver.solve(
                    question=question,
                    image_path=image_path,
                    output_types=self.output_types,
                    max_steps=self.max_steps,
                    max_time=self.max_time,
                    max_tokens=self.max_tokens
                )
                result = agent_result
                if agent_result.get("error_executions"):
                    error_executions = {"pid": index, "question": question, "image_path": image_path, "expected_answer": answer, "agent_name": agent_name, "error_executions": agent_result.get("error_executions", None) }
                    json_dir = os.path.join(self.error_json_dir)
                    os.makedirs(json_dir, exist_ok=True)
                    with open(os.path.join(json_dir, f"output_{index}.json"), "w") as f:
                        json.dump(error_executions, f, indent=4, default=str)

                agent_results[agent_name] = agent_result

                if self.verbose:
                    print(f"✅ {agent_name} completed successfully")
                    if 'final' in agent_result:
                        print(f"Final answer: {agent_result['final_output']}")
                    elif 'base' in agent_result:
                        print(f"Base answer: {agent_result['base_output']}")
                    elif 'direct' in agent_result:
                        print(f"Direct answer: {agent_result['direct_output']}")
                    elif 'full' in agent_result:
                        print(f"Full answer: {agent_result['full_answer']}")
                    else:
                        print(f"{self.output_types[0]}_output: {agent_result[f'{self.output_types[0]}_output']}")
                
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "agent_name": agent_name,
                    "question": question,
                    "image_path": image_path,
                    "result": result,
                    "traceback": traceback.format_exc()
                }
                print(f"Error traceback: {traceback.format_exc()}")
                agent_results[agent_name] = error_result
                
                if self.verbose:
                    print(f"❌ {agent_name} failed: {e}")
        results[f"{self.output_types[0]}_output"] = agent_results

        # Save results
        self._save_results(results, index)

        # Print summary
        self._print_summary(results, index)


    def _save_results(self, results: Dict[str, Any], index: int):
        """Save results to JSON file"""
        # Create output directory
        json_dir = os.path.join(self.output_json_dir)
        os.makedirs(json_dir, exist_ok=True)
        
        # Helper function to add metadata to agent result
        def add_metadata_to_result(agent_result):
            return {
                "pid": results["pid"],
                "query": results["query"],
                "image": results["image"],
                "expected_answer": results["expected_answer"],
                **agent_result
            }
        
        # If only one agent, save in simple format
        if len(results[f"{self.output_types[0]}_output"]) == 1:
            agent_name = list(results[f"{self.output_types[0]}_output"].keys())[0]
            agent_result = results[f"{self.output_types[0]}_output"][agent_name]
            
            # Save in original format: output_{index}.json
            simple_file = os.path.join(json_dir, f"output_{index}.json")
            with open(simple_file, 'w') as f:
                json.dump(add_metadata_to_result(agent_result), f, indent=4, default=str)
            print(f"\n==>Results saved to: {simple_file}")
            
        else:
            # Save individual agent results
            for agent_name, agent_result in results[f"{self.output_types[0]}_output"].items():
                agent_file = os.path.join(json_dir, f"output_{index}_{agent_name}.json")
                with open(agent_file, 'w') as f:
                    json.dump(add_metadata_to_result(agent_result), f, indent=4, default=str)
                if self.verbose:
                    print(f"Saved {agent_name} results to: {agent_file}")

    def _print_summary(self, results: Dict[str, Any], index: int):
        """Print summary of results"""
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR PROBLEM {index}")
        print(f"{'='*80}")
        
        
        print(f"Expected Answer: {results['expected_answer']}")

        agent_outputs = results[f"{self.output_types[0]}_output"]

        def extract_output(result: Dict[str, Any]) -> str:
            preferred_keys = [
                "final_output",
                "direct_output",
                "full_answer",
                *(f"{otype}_output" for otype in self.output_types)
            ]
            for key in preferred_keys:
                if key in result and isinstance(result[key], str):
                    value = result[key]
                    return value[:100] + "..." if len(value) > 100 else value
            return str(result)[:100] + "..."

        if isinstance(agent_outputs, dict):
            for agent_name, agent_result in agent_outputs.items():
                formatted = extract_output(agent_result if isinstance(agent_result, dict) else {"value": agent_result})
                print(f"  {agent_name}: {formatted}")
        else:
            formatted = extract_output(agent_outputs if isinstance(agent_outputs, dict) else {"value": agent_outputs})
            print(f"  Output: {formatted}")
        print(f"\n{'='*80}")


def parse_arguments():
    try:
        parser = argparse.ArgumentParser(description="Run multi-agent benchmark evaluation with specified parameters.")
        parser.add_argument("--llm_engine_name", default="gpt-4o-mini", help="LLM engine name.")
        parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
        parser.add_argument("--task", default="minitoolbench", help="Task to run.")
        parser.add_argument("--data_file", default="data/data.json", help="Data file to run.")
        parser.add_argument("--task_description", default="", help="Task description.")
        parser.add_argument(
            "--output_types",
            default="base,final,direct,full",
            help="Comma-separated list of required outputs (base,final,direct,full)"
        )
        parser.add_argument("--enable_faiss_retrieval",default=False, help="Enable FAISS retrieval to retrieve tools more efficiently.")
        parser.add_argument("--enabled_tools", default="", help="List of enabled tools.")
        parser.add_argument("--agents", default="opentools,octotools,react,chain_of_thought,zero_shot", help="Comma-separated list of agents to use.")
        parser.add_argument("--index", type=int, default=0, help="Index of the problem in the benchmark file.")
        parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
        parser.add_argument("--output_json_dir", default="results", help="Path to output JSON directory.")
        parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
        parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
        parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
        parser.add_argument("--list_agents", action="store_true", help="List available agents and exit.")
        parser.add_argument("--error_json_dir", default="error_json", help="Path to error JSON directory.")
        return parser.parse_args()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        raise e


def main(args):
    if args.list_agents:
        print("Available agents:")
        for agent_info in list_agents():
            print(f"  {agent_info['name']}: {agent_info['description']}")
        return

    # Parse agents list
    agents = args.agents.split(",") if args.agents else ["react"]

    # Instantiate MultiAgentBenchmarkSolver
    solver = MultiAgentBenchmarkSolver(
        agents=agents,
        llm_engine_name=args.llm_engine_name,
        task=args.task,
        data_file=args.data_file,
        task_description=args.task_description,
        output_types=args.output_types,
        index=args.index,
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir,
        enabled_tools=args.enabled_tools,
        error_json_dir=args.error_json_dir,
        enable_faiss_retrieval=args.enable_faiss_retrieval
    )

    # Solve the task or problem
    solver.solve()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)