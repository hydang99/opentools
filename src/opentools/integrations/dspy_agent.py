"""DSPy agent program and prompt optimization for OpenTools callables."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional

import dspy

from .dspy import as_callable


def _finish(answer: str) -> str:
    """Conclude the trajectory and return the final answer."""
    return answer


def _function_metadata(function: Callable[..., Any]) -> Dict[str, str]:
    return {
        "function_name": function.__name__,
        "arguments": str(inspect.signature(function)),
        "docstring": inspect.getdoc(function) or "No documentation provided.",
    }


class OpenToolsDSPyAgent(dspy.Module):
    """A compilable DSPy tool-selection policy backed by OpenTools callables.

    DSPy optimizers modify the ``policy`` predictor's instructions and examples;
    they do not modify or claim to improve the underlying tool implementation.
    """

    def __init__(
        self,
        tool_names: Iterable[str],
        *,
        max_steps: int = 5,
        max_risk: str = "low",
        tools_root: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        names = list(tool_names)
        if not names:
            raise ValueError("At least one OpenTools tool name is required")

        functions = {
            name: as_callable(name, tools_root=tools_root, max_risk=max_risk)
            for name in names
        }
        functions["finish"] = _finish
        self.functions = functions
        self.tools = {
            name: _function_metadata(function)
            for name, function in self.functions.items()
        }
        self.max_steps = max_steps
        policy_instructions = instructions or (
            "Select and call the available tools to answer the question. Inspect each "
            "tool's arguments before selecting it. Use observations from earlier steps, "
            "and select finish with a concise final answer when the task is complete."
        )
        signature = dspy.Signature(
            "question, trajectory, tools -> next_selected_tool, arguments: dict[str, Any]",
            policy_instructions,
        )
        self.policy = dspy.ChainOfThought(signature)

    def forward(self, question: str):
        trajectory: List[Dict[str, Any]] = []
        final_answer: Any = ""

        for _ in range(self.max_steps):
            prediction = self.policy(
                question=question,
                trajectory=trajectory,
                tools=self.tools,
            )
            selected = str(prediction.next_selected_tool).strip('"').strip("'")
            arguments = prediction.arguments
            if not isinstance(arguments, dict):
                observation = {
                    "return_value": None,
                    "error": "Tool arguments must be a JSON object.",
                }
            elif selected not in self.functions:
                observation = {
                    "return_value": None,
                    "error": f"Unknown tool selected: {selected}",
                }
            else:
                try:
                    value = self.functions[selected](**arguments)
                    observation = {"return_value": value, "error": None}
                except Exception as exc:
                    observation = {
                        "return_value": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            trajectory.append(
                {
                    "reasoning": prediction.reasoning,
                    "selected_tool": selected,
                    "arguments": arguments,
                    **observation,
                }
            )
            if selected == "finish" and observation["error"] is None:
                final_answer = observation["return_value"]
                break

        return dspy.Prediction(answer=final_answer, trajectory=trajectory)


def optimize_with_simba(
    agent: OpenToolsDSPyAgent,
    *,
    trainset: List[Any],
    metric: Callable[..., Any],
    batch_size: int = 32,
    max_steps: int = 8,
    max_demos: int = 4,
    seed: int = 0,
    num_threads: Optional[int] = None,
    prompt_model: Any = None,
):
    """Compile an OpenTools DSPy agent using real examples and a real metric."""
    if not trainset:
        raise ValueError("trainset must contain at least one dspy.Example")
    if len(trainset) < batch_size:
        raise ValueError(
            f"trainset must contain at least batch_size examples: "
            f"{len(trainset)} < {batch_size}"
        )
    optimizer = dspy.SIMBA(
        metric=metric,
        bsize=batch_size,
        max_steps=max_steps,
        max_demos=max_demos,
        num_threads=num_threads,
        prompt_model=prompt_model,
    )
    return optimizer.compile(agent, trainset=trainset, seed=seed)


__all__ = ["OpenToolsDSPyAgent", "optimize_with_simba"]
