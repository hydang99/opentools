"""
Centralized display and logging system for OpenTools agents.
Provides rich formatting, step tracking, and unified output across all agents.
"""

import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import json


class LogLevel(Enum):
    """Log levels for different types of messages"""
    DEBUG = "DEBUG"
    INFO = "INFO"  
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    STEP = "STEP"
    RESULT = "RESULT"


class DisplayManager:
    """
    Centralized display manager for agent outputs.
    Handles formatting, step tracking, and rich display across all agents.
    """
    
    def __init__(self, agent_name: str, verbose: bool = True, use_colors: bool = True):
        self.agent_name = agent_name
        self.verbose = verbose
        self.use_colors = use_colors
        self.current_step = 0
        self.step_start_time = None
        self.total_start_time = time.time()
        self.step_history = []
        
        # Color codes for different log levels
        self.colors = {
            LogLevel.DEBUG: "\033[90m",      # Gray
            LogLevel.INFO: "\033[94m",       # Blue  
            LogLevel.SUCCESS: "\033[92m",    # Green
            LogLevel.WARNING: "\033[93m",    # Yellow
            LogLevel.ERROR: "\033[91m",      # Red
            LogLevel.STEP: "\033[96m",       # Cyan
            LogLevel.RESULT: "\033[95m",     # Magenta
            "RESET": "\033[0m"               # Reset
        } if use_colors else {level: "" for level in LogLevel}
        self.colors["RESET"] = "\033[0m" if use_colors else ""
    
    def _format_message(self, message: str, level: LogLevel, prefix: str = None) -> str:
        """Format a message with colors and prefixes"""
        if not self.verbose:
            return ""
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.colors.get(level, "")
        reset = self.colors["RESET"]
        
        if prefix:
            formatted = f"{color}[{timestamp}][{self.agent_name}][{level.value}] {prefix}: {message}{reset}"
        else:
            formatted = f"{color}[{timestamp}][{self.agent_name}][{level.value}] {message}{reset}"
        
        return formatted
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO, prefix: str = None):
        """Basic logging method with enhanced formatting"""
        formatted = self._format_message(message, level, prefix)
        if formatted:
            print(formatted)
    
    def start_step(self, step_description: str, step_data: Dict[str, Any] = None) -> int:
        """Start a new step and return the step number"""
        self.current_step += 1
        self.step_start_time = time.time()
        
        # Format step header
        step_header = f"{'='*60}"
        step_title = f"Agent Step {self.current_step}: {step_description}"
        step_footer = f"{'='*60}"
        
        color = self.colors[LogLevel.STEP]
        reset = self.colors["RESET"]
        
        if self.verbose:
            print(f"\n{color}{step_header}")
            print(f"{step_title}")
            print(f"{step_footer}{reset}")
        
        # Store step info
        step_info = {
            "step_number": self.current_step,
            "description": step_description,
            "start_time": self.step_start_time,
            "data": step_data or {}
        }
        self.step_history.append(step_info)
        
        return self.current_step
    
    def end_step(self, result: str = None, success: bool = True):
        """End the current step with optional result"""
        if not self.step_start_time:
            return
            
        duration = time.time() - self.step_start_time
        
        # Update step history
        if self.step_history:
            self.step_history[-1].update({
                "end_time": time.time(),
                "duration": duration,
                "result": result,
                "success": success
            })
        
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        status = "✓" if success else "✗"
        
        if result:
            self.log(f"{status} Step {self.current_step} completed in {duration:.2f}s: {result}", level)
        else:
            self.log(f"{status} Step {self.current_step} completed in {duration:.2f}s", level)
        
        self.step_start_time = None
    
    def log_thought(self, thought: str):
        """Log a reasoning/thought step"""
        self.log(f"💭 {thought}", LogLevel.INFO, "Thought")
    
    def log_action(self, action: str, tool_name: str = None):
        """Log an action being taken"""
        if tool_name:
            self.log(f"🔧 Using {tool_name}: {action}", LogLevel.INFO, "Action")
        else:
            self.log(f"⚡ {action}", LogLevel.INFO, "Action")
    
    def log_observation(self, observation: str):
        """Log an observation/result"""
        self.log(f"👁️ {observation}", LogLevel.RESULT, "Observation")
    
    def log_tool_command(self, command: str):
        """Log a tool command being executed"""
        self.log(f"⚙️ {command}", LogLevel.DEBUG, "Command")
    
    def log_error(self, error: str, exception: Exception = None):
        """Log an error with optional exception details"""
        if exception:
            self.log(f"❌ {error}: {str(exception)}", LogLevel.ERROR)
        else:
            self.log(f"❌ {error}", LogLevel.ERROR)
    
    def log_warning(self, warning: str):
        """Log a warning message"""
        self.log(f"⚠️ {warning}", LogLevel.WARNING)
    
    def log_final_answer(self, answer: str):
        """Log the final answer with special formatting"""
        color = self.colors[LogLevel.SUCCESS]
        reset = self.colors["RESET"]
        
        if self.verbose:
            print(f"\n{color}{'='*60}")
            print(f"🎯 FINAL ANSWER")
            print(f"{'='*60}{reset}")
            print(f"{color}{answer}{reset}")
            print(f"{color}{'='*60}{reset}\n")
    
    def display_progress_summary(self):
        """Display a summary of all steps taken"""
        if not self.step_history:
            return
            
        total_time = time.time() - self.total_start_time
        
        color = self.colors[LogLevel.INFO]
        reset = self.colors["RESET"]
        
        if self.verbose:
            print(f"\n{color}📊 EXECUTION SUMMARY")
            print(f"{'='*50}")
            print(f"Agent: {self.agent_name}")
            print(f"Total Steps: {len(self.step_history)}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*50}")
            
            for i, step in enumerate(self.step_history, 1):
                status = "✓" if step.get("success", True) else "✗"
                duration = step.get("duration", 0)
                print(f"{status} Step {i}: {step['description']} ({duration:.2f}s)")
            
            print(f"{'='*50}{reset}\n")
    
    def get_trace_formatted(self, include_metadata: bool = True) -> str:
        """Get a formatted trace of all steps for inclusion in prompts"""
        if not self.step_history:
            return ""
        
        trace_parts = []
        
        for step in self.step_history:
            step_num = step["step_number"]
            desc = step["description"]
            
            trace_parts.append(f"Agent Step {step_num}: {desc}")
            
            if include_metadata and step.get("result"):
                trace_parts.append(f"Result: {step['result']}")
            
            if include_metadata and step.get("duration"):
                trace_parts.append(f"Duration: {step['duration']:.2f}s")
        
        return "\n\n".join(trace_parts)
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data for analysis"""
        return {
            "agent_name": self.agent_name,
            "total_execution_time": time.time() - self.total_start_time,
            "total_steps": len(self.step_history),
            "step_history": self.step_history,
            "timestamp": datetime.now().isoformat()
        }


class AgentDisplayMixin:
    """
    Mixin class to add enhanced display capabilities to any agent.
    Should be used alongside BaseAgent.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_manager = None
    
    def get_display_manager(self) -> DisplayManager:
        """Get or create the display manager for this agent"""
        # Handle cases where _display_manager might not exist yet
        if not hasattr(self, '_display_manager') or self._display_manager is None:
            self._display_manager = DisplayManager(
                agent_name=self.get_agent_name(),
                verbose=getattr(self, 'verbose', True),
                use_colors=True
            )
        return self._display_manager
    
    # Enhanced logging methods
    def log_step_start(self, description: str, data: Dict[str, Any] = None) -> int:
        """Start a new step with description"""
        return self.get_display_manager().start_step(description, data)
    
    def log_step_end(self, result: str = None, success: bool = True):
        """End the current step"""
        self.get_display_manager().end_step(result, success)
    
    def log_thought(self, thought: str):
        """Log a thought/reasoning step"""
        self.get_display_manager().log_thought(thought)
    
    def log_action(self, action: str, tool_name: str = None):
        """Log an action"""
        self.get_display_manager().log_action(action, tool_name)
    
    def log_observation(self, observation: str):
        """Log an observation"""
        self.get_display_manager().log_observation(observation)
    
    def log_tool_command(self, command: str):
        """Log a tool command"""
        self.get_display_manager().log_tool_command(command)
    
    def log_final_answer(self, answer: str):
        """Log the final answer with special formatting"""
        self.get_display_manager().log_final_answer(answer)
    
    def log_error(self, error: str, exception: Exception = None):
        """Log an error"""
        self.get_display_manager().log_error(error, exception)
    
    def log_warning(self, warning: str):
        """Log a warning"""
        self.get_display_manager().log_warning(warning)
    
    def display_summary(self):
        """Display execution summary"""
        self.get_display_manager().display_progress_summary()
    
    def get_formatted_trace(self) -> str:
        """Get formatted trace for prompts"""
        return self.get_display_manager().get_trace_formatted()
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export session data"""
        return self.get_display_manager().export_session_data()
    
    # Backward compatibility - enhance the original log method
    def log(self, message: str, level: str = "INFO"):
        """Enhanced version of the original log method"""
        # Convert string level to LogLevel enum
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "SUCCESS": LogLevel.SUCCESS,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "STEP": LogLevel.STEP,
            "RESULT": LogLevel.RESULT
        }
        
        log_level = level_map.get(level.upper(), LogLevel.INFO)
        self.get_display_manager().log(message, log_level)