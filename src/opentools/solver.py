"""
OpenTools Shared Solver

This module provides a unified interface for solving tasks using different agents.
Users can select which agent to use and the solver will handle the execution.
"""

import argparse
import time
import json
from typing import Optional, Dict, Any
from datetime import datetime

from .agents import create_agent, list_agents, BaseAgent


class UnifiedSolver:
    """
    Unified solver that can work with any registered agent.
    Provides a consistent interface regardless of the underlying agent implementation.
    """
    
    def __init__(self, 
                 agent_name: str = "opentools",
                 llm_engine_name: str = "gpt-4o-mini",
                 verbose: bool = True,
                 **agent_kwargs):
        """
        Initialize the unified solver.
        
        Args:
            agent_name: Name of the agent to use
            llm_engine_name: LLM engine name
            verbose: Whether to print verbose output
            **agent_kwargs: Additional arguments to pass to the agent
        """
        self.agent_name = agent_name
        self.llm_engine_name = llm_engine_name
        self.verbose = verbose
        self.agent_kwargs = agent_kwargs
        
        # Create the agent
        self.agent = self._create_agent()
        
        if self.verbose:
            print(f"UnifiedSolver initialized with agent: {self.agent.get_agent_name()}")
            print(f"Agent description: {self.agent.get_agent_description()}")
    
    def _create_agent(self) -> BaseAgent:
        """Create the agent instance"""
        try:
            return create_agent(
                agent_name=self.agent_name,
                llm_engine_name=self.llm_engine_name,
                verbose=self.verbose,
                **self.agent_kwargs
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating agent '{self.agent_name}': {e}")
                print("Available agents:")
                for agent_info in list_agents():
                    print(f"  - {agent_info['name']}: {agent_info['description']}")
            raise
    
    def solve(self, 
              question: str,
              image_path: Optional[str] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Solve a question using the selected agent.
        
        Args:
            question: The question to solve
            image_path: Optional path to an image
            **kwargs: Additional arguments to pass to the agent's solve method
            
        Returns:
            Dictionary containing the solution and metadata
        """
        start_time = time.time()
        
        try:
            # Add unified solver metadata
            result = self.agent.solve(question=question, image_path=image_path, **kwargs)
            
            # Add unified solver metadata
            result.update({
                "solver_type": "UnifiedSolver",
                "agent_used": self.agent_name,
                "llm_engine": self.llm_engine_name,
                "total_execution_time": time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "solver_type": "UnifiedSolver", 
                "agent_used": self.agent_name,
                "llm_engine": self.llm_engine_name,
                "question": question,
                "image_path": image_path,
                "total_execution_time": time.time() - start_time
            }
            
            if self.verbose:
                print(f"Error in UnifiedSolver: {e}")
            
            return error_result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent"""
        return self.agent.get_agent_info()
    
    @staticmethod
    def list_available_agents():
        """List all available agents"""
        return list_agents()


def main():
    """CLI interface for the unified solver"""
    parser = argparse.ArgumentParser(description="OpenTools Unified Solver")
    parser.add_argument("question", help="Question to solve")
    parser.add_argument("--agent", default="opentools", help="Agent to use")
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM engine")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--output-types", default="final", help="Output types")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps")
    parser.add_argument("--max-time", type=int, default=300, help="Maximum time")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--list-agents", action="store_true", help="List available agents")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--enable-faiss-retrieval", action="store_true", help="Enable FAISS retrieval")
    
    args = parser.parse_args()
    
    if args.list_agents:
        print("Available agents:")
        for agent_info in list_agents():
            print(f"  {agent_info['name']}: {agent_info['description']}")
        return
    
    # Create solver
    solver = UnifiedSolver(
        agent_name=args.agent,
        llm_engine_name=args.llm,
        verbose=args.verbose
    )
    
    # Solve
    solve_kwargs = {
        "output_types": args.output_types,
        "max_steps": args.max_steps, 
        "max_time": args.max_time,
        "enable_faiss_retrieval": args.enable_faiss_retrieval
    }
    
    if args.cache_dir:
        solve_kwargs["root_cache_dir"] = args.cache_dir
    if args.enable_faiss_retrieval:
        solve_kwargs["enable_faiss_retrieval"] = True
    
    result = solver.solve(
        question=args.question,
        image_path=args.image,
        **solve_kwargs
    )
    
    # Output result
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()