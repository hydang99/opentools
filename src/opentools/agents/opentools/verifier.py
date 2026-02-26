from typing import Any
from pydantic import BaseModel, Field

class LocalVerify(BaseModel):
   verification: bool = Field(
      description="True if the tool result contains information that supports or is relevant to the current sub_problem, even if it doesn't directly answer the query. False only if the result is completely irrelevant, empty, or contains errors."
   ),
   summary_result: str = Field(
      description="A comprehensive summary of the important results extracted from the tool output. If verification=True, include ALL important information: URLs, paper titles/IDs, file paths, key data points, extracted values, and any other details that next steps might need. If verification=False, return an empty string."
   )

   reason: str = Field(
      description="If verification=False, a short reason why it failed. If verification=True, return an empty string."
   )
   suggestion: str = Field(
      description="If verification=False, suggest specific actionable fix for the next attempt (what to change/add/redo). If verification=True, return an empty string."
   )

class Verifier:
   def __init__(self, llm_engine: Any):
      self.llm_engine = llm_engine

      self.sys_prompt_local_verification = """
You are the Answer Verifier in a multi-step agent system.

Instructions:
- You judge whether the tool result contains information that SUPPORTS or is RELEVANT to the current query.
- Be LENIENT: The result does NOT need to directly answer the query. If it contains any relevant information, data, or resources that could help address the query, return verification=True.
- However, if the result is completely irrelevant, empty, or contains errors, return verification=False.
- When in doubt (borderline cases), set verification=True to allow the system to use the information.

You are given:
- Current query (the target for this step)
- The agent and tool calls used to generate the solution
- The solution/result from the tool execution

Your job:

** When to return verification=True (BE LENIENT): **
- The result contains ANY information, data, or resources relevant to the query.
- The result includes URLs, file paths, paper titles, extracted data, or other resources that could be useful.
- The result partially addresses the query or provides supporting information.
- The result is relevant to the query domain even if it doesn't directly answer it.
- The tool executed successfully and returned any meaningful output related to the query topic.

** When to return verification=False (ONLY in clear failure cases): **
- If the tool execution failed with errors and return result is None or not useful.
- The result is completely empty or contains no meaningful content.
- The result is completely irrelevant to the query (wrong domain/topic entirely).
- The tool execution failed with errors and returned no useful information.
- The result contains only error messages with no recoverable information.

** summary_result field (CRITICAL - only when verification=True): **
- Extract and include ALL important information from the tool result that next steps might need:
  * URLs, file paths, download links, paper PDF URLs
  * Paper titles, arXiv IDs, DOIs, entry IDs
  * Extracted values, data points, measurements, counts
  * File names, document paths, image paths
  * Key identifiers, IDs, references
  * Important metadata, dates, categories
  * Any structured data or lists returned
- Format the summary clearly so it can be used by subsequent steps without needing to access the raw result again.
- Include enough context that the next step understands what was retrieved and can use the information effectively.
- DO NOT summarize too aggressively - preserve important details like exact URLs, IDs, and specific values.

** reason and suggestion fields: **
- If verification=True:
  - set reason and suggestion to empty string.
- If verification=False:
  - reason: short, concrete reason why it failed (e.g., "empty result", "tool execution error", "completely irrelevant to query").
  - suggestion: specific actionable fix for the next attempt (what to change/add/redo).
      """

   def generate_prompt_answer_verification(self, query: str, agent: str, tool_calls: str, result: str):
      prompt_answer_verification = f"""
      Context:
      Query: {query}
      Agent: {agent}
      Tool calls: {tool_calls}
      Result: {result}
      """
      return prompt_answer_verification


   def verify_tool_result(self, query, result, agent, tool_calls):
      input_prompt = self.generate_prompt_answer_verification(query = query, agent = agent, tool_calls = tool_calls, result = result)
      response = self.llm_engine.generate(query = input_prompt, system_prompt = self.sys_prompt_local_verification, response_format = LocalVerify)
      return response