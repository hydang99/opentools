# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/generalist_solution_generator/tool.py
import os, sys, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class Generalist_Solution_Generator_Tool(BaseTool):
    """Generalist_Solution_Generator_Tool
    ---------------------
    Purpose:
        A versatile AI-powered tool that generates step-by-step solutions for general queries and tasks. Can process both text prompts, images and files, providing comprehensive responses using OpenAI's language models. Ideal for general problem-solving, analysis, and reasoning tasks that don't require specialized tools.

    Core Capabilities:
        - Generates step-by-step solutions for general queries
        - Processes both text, image and file inputs
        - Provides reasoning and analysis
        - Handles various types of questions and tasks

    Intended Use:
        Use this tool when you need to generate step-by-step solutions for general queries and tasks, including text, image and file inputs.

    Limitations:
        - May not handle complex questions and tasks
        - Requires a valid OpenAI API key and internet connectivity

    """
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Generalist_Solution_Generator_Tool",
            description="""A versatile AI-powered tool that generates step-by-step solutions for general queries and tasks. 
            Can process both text prompts, images and files, providing comprehensive responses using OpenAI's language models. 
            Ideal for general problem-solving, analysis, and reasoning tasks that don't require specialized tools. 
            CAPABILITIES: Generates step-by-step solutions for general queries, processes both text, image and file inputs, 
            provides reasoning and analysis, handles various types of questions and tasks. SYNONYMS: general AI assistant, 
            problem solver, reasoning tool, AI analyzer, general purpose AI, solution generator, AI helper, 
            reasoning assistant. EXAMPLES: 'Explain this concept step by step', 'Analyze the mood of this image', 
            'Solve this problem with reasoning', 'Summarize this information'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query/problem that needs to be solved. Be direct and concise—do not include unnecessary extra information. Focus only on the problem that needs to be solved.",
                    },
                    "image": {
                        "type": "string",
                        "description": "Optional path to an image file for multimodal analysis (default: None)"
                    },
                    "file": {
                        "type": "string",
                        "description": "Optional path to a file for analysis (default: None)"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="ai_assistance",
            tags=["ai_assistant", "problem_solving", "reasoning", "multimodal", "general_purpose", "analysis", "solution_generator"],
            limitation="May provide hallucinated or incorrect responses, not suitable for complex multi-step reasoning tasks, requires OpenAI API key, image processing limited to supported formats, responses should be verified for accuracy",
            agent_type="General-Agent",
            accuracy= 'Rely on the model used',
            demo_commands= {    
                "command": "reponse = tool.run(query='What is the capital of France?', image='test.jpg')",    
                "description": "Generate a solution for a general query and image"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )

    def run(self, query, image=None, file=None):        
        try:
            file_path = image if image else file
            if file_path:
                # Clean up image path
                if not (file_path.startswith("http://") or file_path.startswith("https://") or os.path.exists(file_path)):
                    file_path = file_path.replace("\\", "/").replace("\\", "/")     
                    if not os.path.isfile(file_path):
                        return {"error": "Error: Invalid file path.", "success": False, "error_type": "invalid_file_path"}
                    if not os.path.isabs(file_path):
                        file_path = os.path.abspath(file_path)
                if image:
                    response = self.llm_engine.generate(query, image=file_path)
                else:
                    response = self.llm_engine.generate(query, file=file_path)

            else:
                response = self.llm_engine.generate(query)
            if isinstance(response, dict):
                response = response.get('text')
            else:
                response = str(response)
            return {"result": response, "success": True, "metadata": {
                "prompt_tokens": self.llm_engine.get_token_usage()
            }}
                
        except Exception as e:
            return {"error": f"Error generating response: {str(e)}", "success": False, "error_type": "generalist_solution_generator_failed", "traceback": traceback.format_exc()}
    
if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/generalist_solution_generator
    python tool.py
    """
    try:
        tool = Generalist_Solution_Generator_Tool()
        tool.embed_tool()
        print(tool.run(query= "How many pages is the file?", file= r"/home/daoqm/opentools_gaia_running/opentools/src/opentools/Benchmark/downloads/math.pdf"))

    except Exception as e:
        print(f"Error: {e}")    