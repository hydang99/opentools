from typing import Any, Dict
class Global_Memory:
    def __init__(self, query: str):
        self.query = query
        self.step_count = 0
        self.actions = {}

    def update_step(self, sub_problem: str, answer: str):
        self.actions[f"Agent_Step_{self.step_count}"] = {
            "sub_problem": sub_problem,
            "answer": answer,
        }
        self.step_count += 1
    
    def update_answer(self, answer: str):
        self.actions.update({
            "final_answer": answer,
        })

    def get_actions(self):
        return self.actions


class Local_Memory:
    def __init__(self):
        self.query = None
        self.sub_agent = None
        self.step_count = -1
        self.actions = {}

    def get_action(self):
        return self.actions

    def update_memory_reasoning(self, sub_problem: str, sub_agent: str, supporting_documents: str):
        self.step_count += 1
        self.actions[f"Agent_Step_{self.step_count}"] = {
            "sub_problem": sub_problem,
            "sub_agent": sub_agent,
            "supporting_documents": supporting_documents,
        }

    def update_insufficient_info_reason(self, reason: str):
        """Update the reason why there is insufficient information to answer the query."""
        # Initialize step_count if needed
        if self.step_count == -1:
            self.step_count = 0
        # Create entry if it doesn't exist
        if self.step_count not in self.actions:
            self.actions[f"Agent_Step_{self.step_count}"] = {}
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "insufficient_info_reason": reason,
        })
    def update_tool_error(self, tool_error: str):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "tool_error": tool_error,
        })
    def update_failure_source(self, failure_source: str):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "failure_source": failure_source,
        })
    

    def update_tool_calls(self, tool_calls: Any):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "tool_calls": tool_calls,
        })
    def update_tool_result(self, tool_result: Any):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "tool_result": tool_result,
        })
    def update_tool_error(self, tool_error: str):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "tool_error": tool_error,
        })
    def update_error(self, error: str):
        # Handle case where step_count is -1 (error before first step)
        if self.step_count == -1:
            self.step_count = 0
        # Create entry if it doesn't exist
        if f"Agent_Step_{self.step_count}" not in self.actions:
            self.actions[f"Agent_Step_{self.step_count}"] = {}
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "error": error,
        })

    def update_verification_failure(self, verification_failure: str, reason: str, suggestion: str):
        self.actions[f"Agent_Step_{self.step_count}"].update({
            "verification_failure": verification_failure,
            "reason": reason,
            "suggestion": suggestion,
        })
    def get_actions(self):
        return self.actions