from typing import Any
from pydantic import BaseModel, Field

class NextStep(BaseModel):
    sub_problem: str = Field(
        description="One specific, actionable next sub-task that advances the original user query. Must be a single step (not a full plan), written so a domain sub_agent can act immediately using verified memory. Must not mention tool names or tool calls. If stop is True, set sub_problem to None."   
    )
    sub_agent: str = Field(
        description= "The single best domain sub_agent to handle sub_problem. Must be exactly one of the provided sub_agent_list values (do not invent names). Prefer the most specific domain agent; use a general agent only as fallback if it exists in sub_agent_list. If stop is True, set sub_agent to None."
        , choices=["Visual-Agent", "Browser_Extraction-Agent", "General-Agent", "File_Extraction-Agent"
            ,"Media-Agent", "Search-Agent", "Download_File-Agent", "Puzzle-Agent","Mathematics-Agent"]    
    ),
    supporting_documents: str | None = Field(
        description= "Single supporting document like image, file, url, etc if exists and relevant to the sub_problem. Only include bare link without any other text or explanation. If no supporting documents are available, set supporting_documents to None. If stop is True, set supporting_documents to None."
    ),
    stop: bool = Field(
        description= "Set stop=True only if verified memory has sufficient information to answer the original user query. Otherwise set stop=False."
    ),
    answer: str = Field(
        description= "If stop is True, include the answer to the original user query. Otherwise set answer to None."
    ),



class RetryStep(BaseModel):
    sub_problem: str = Field(
        description= "One specific, actionable corrective next sub-task that advances the original user query after a failure. Must be a single step (not a full plan), written so a domain sub_agent can act immediately using ONLY verified ground-truth information in memory. Must not mention tool names or tool calls."
    )
    sub_agent: str = Field(
        description= "The single best domain sub_agent to handle sub_problem. Must be exactly one of the provided sub_agent_list values (do not invent names). Prefer the most specific domain agent; use a general agent only as fallback if it exists in sub_agent_list."
    , choices=["Visual-Agent", "Browser_Extraction-Agent", "General-Agent", "File_Extraction-Agent"
            ,"Media-Agent", "Search-Agent", "Download_File-Agent", "Puzzle-Agent","Mathematics-Agent"]   
    ),
    supporting_documents: str | None = Field(
        description= "Single supporting document like image, file, url, etc if exists and relevant to the sub_problem. Only include bare link without any other text or explanation. If no supporting documents are available, set supporting_documents to None."
    ),

    failure_source: str = Field(
        description= "Which component most likely caused the last failure. Must be one of: 'sub_problem' (the goal/objective was wrong/underspecified/ambiguous/wrong target), 'sub_agent' (wrong domain agent chosen), 'tools' (goal and agent were reasonable but tool usage/arguments/retrieval/evidence handling failed)."
        , choices=["sub_problem", "sub_agent", "tools"]
    )
    
class Reasoning:
    def __init__(self, llm_engine: Any):
       self.llm_engine = llm_engine
       self.sys_prompt_next_step = """
You are the Next-Step Planner in a multi-step agent system with multiple domain sub_agents. 

Instructions:
- You only choose the best next sub_problem and the single best sub_agent to handle it and decide when there is enough information to answer the original query.
- You do NOT execute tools, do NOT generate tool calls.
- Avoid repetition. Don't repeat the same last step if it is already successful.

You are given:
- Original user query
- Available sub_agents
- Memory of the system (the steps taken so far)
- (Optional) Reason why there is still insufficient information to answer the original query.

Memory rule:
- Memory contains ONLY verified ground-truth information from successful steps.

Your job:

** sub_problem selection: **
- Break the task into small enough sub_problems that can be solved by a single sub_agent, unless the General-Agent is being used—in that case, it is acceptable for the General-Agent to handle a larger or full problem directly.
- Don't try to generate a too large plan for a single agent to handle.
- Include necessary information like image, file, url, etc if exists and relevant to the sub_problem.
- The sub_problem must be specific and actionable so that the sub_agent can act without guessing what to do.
- Do not generate sub_problems that require gathering excessive or unnecessary background information. Focus only on the next direct step needed for answering the original user query.
- Avoid creating overly broad or lengthy sub_problems. Be concise and target only the specific information required to make progress.

** sub_agent selection: **
- Choose the single best sub_agent from the current available sub_agents list that is most suitable for solving the sub_problem.
- The sub_agent must be one of the available sub_agents (do not invent names).
- Prefer the most specific domain agent. Use "General-Agent" sub_agents ONLY if there is no specific sub_agent that can handle the sub_problem or the other sub_agents are already failed.

** Detailed agent selection guidelines: **
- **Visual-Agent**: Use for extracting information/text from images and answering simple questions about images. Choose when the sub_problem involves analyzing visual content, reading text from images, or understanding image-based information.
- **Browser_Extraction-Agent**: Use for extracting information or text directly from HTML web pages/URLs (landing pages). Best used after Search-Agent identifies relevant URLs. Choose when the sub_problem requires extracting content from a specific webpage or URL. Do NOT choose this agent for URLs that end with file types such as `.pdf`, `.jpg`, or other downloadable/attachment file extensions; use Download_File-Agent (for remote files) and then File_Extraction-Agent instead.
- **File_Extraction-Agent**: Use for extracting information/text from any type of files (PDF, DOCX, CSV, etc.). Only local files are supported. Works best with Download_File-Agent to download files first. Choose when the sub_problem involves reading or extracting data from file formats. 
  - If the task is to **find/count which pages contain a search term/phrase** (e.g., “how many pages mention X”, “list page numbers containing Y”), make the sub_problem explicitly ask for: total page count, the list of page numbers with key phrase, and the count of pages with key phrase (avoid asking for full-document extraction).
  - If the agent fails to extract the information (invalid format or token limit), use General-Agent to deal with the problem.
- **Media-Agent**: Use for extracting information from media files like images, videos, or audio. Choose when the sub_problem involves processing multimedia content beyond simple image analysis.
- **Search-Agent**: Use for searching credible sources of information (Google search, Wikipedia, chemistry search, paper search, news search, etc.). Can only search for results, cannot extract information from search results. Choose when the sub_problem requires finding information sources or URLs, but not extracting content from them.
- **Download_File-Agent**: Use for downloading files from the web. Choose when the sub_problem requires obtaining a file from a URL before it can be processed. If the file has already been downloaded, don't try to download it again.
- **Puzzle-Agent**: Use for solving puzzles like mazes, Rubik's cubes, N-queens, etc. Cannot solve math problems, only puzzle problems. Choose when the sub_problem involves logic puzzles, pathfinding, or game-solving.
- **Mathematics-Agent**: Use for solving mathematics problems like algebra, geometry, calculus, or simple arithmetic calculations. Cannot solve puzzle problems, only pure mathematics problems. Choose when the sub_problem involves mathematical computation or problem-solving.
- **General-Agent**: Use as a fallback when no specific domain agent can handle the sub_problem. Only choose if no specialized agent is available or suitable for the task or the other sub_agents are already failed.
- If all the tools in the specialized sub_agent are already failed and no other sub_agent can handle the sub_problem, choose General-Agent to deal with the problem.
- If supporting_documents is a URL/file path that ends with ".pdf" (or other downloadable/attachment file extensions), you MUST NOT choose Browser_Extraction-Agent. However, if it is a normal URL then you can choose Browser_Extraction-Agent to extract the information from the webpage.

** supporting_documents selection: **
- Choose list of supporting documents from the current available supporting documents list that is most relevant to the sub_problem. If no supporting documents are available, set supporting_documents to None.
- The supporting documents must be a real link to a document that exists and is accessible without any other text or explanation.(Not a fake link or a link to a non-existent document.)

** Stop field: **
- Set stop to True only if there is sufficient information to answer the original query. Otherwise set stop to False.
- If stop is True, set sub_problem, sub_agent and supporting_documents to None.

** Answer field: **
- If stop is True, include the answer to the original user query. Otherwise set answer to None.

       """
       self.sys_prompt_retry = """
You are the Retry Next-Step Planner in a multi-step agent system with multiple domain sub_agents. 

Instructions:
- You only choose the best next corrective sub_problem, the single best sub_agent to handle it, and the failure_source of the last step.
- You do NOT execute tools, do NOT generate tool calls, and do NOT verify results.
- Your goal is to recover from a failed step and continue making progress toward the original user query.

Hard routing rules (MUST FOLLOW):
- If supporting_documents is a URL (or file path) that ends with ".pdf" (or clearly points to a downloadable PDF file), you MUST NOT choose Browser_Extraction-Agent.
  - Choose Download_File-Agent if it is an HTTP/HTTPS URL (to download the PDF locally).
  - Choose File_Extraction-Agent if it is already a local PDF file path (to extract content from the PDF).

You are given:
- Original user query
- Available sub_agents
- Memory of the system (the steps taken so far)
- Failure reports for a few last steps

Failure reports include 2 cases:
- The reason why recent last steps failed and the recent last attempted sub_problems.
- Or the reason why there is still insufficient information to answer the original query.

Memory rule:
- Memory contains ONLY verified ground-truth information from successful steps.

Your job:
** Failure-aware sub_problem selection: **
- Read the failure report carefully and identify the root cause category.
- Your sub_problem must directly respond to the failure_report.
- The corrective sub_problem must be doable by one of the current available sub_agents and must use information from memory as ground truth.
- The sub_problem must be specific and actionable so that the sub_agent can act without guessing what to do.
- Include necessary information like image, file, url, etc if exists and relevant to the sub_problem.
- If the last attempted sub_problem was too complex/too broad, then break the task into small enough sub_problems that can be solved by a single sub_agent, unless the General-Agent is being used—in that case, it is acceptable for the General-Agent to handle a larger or full problem directly.
- Don't try to generate a too large plan for a single agent to handle.
- Do not generate sub_problems that require gathering excessive or unnecessary background information. Focus only on the next direct step needed for answering the original user query.
- Avoid creating overly broad or lengthy sub_problems. Be concise and target only the specific information required to make progress.

** sub_agent selection: **
- Choose the single best sub_agent from the current available sub_agents list that is most suitable for solving the corrective sub_problem.
- The sub_agent must be one of the available sub_agents (do not invent names).
- Prefer the most specific domain agent. Use "General-Agent" sub_agents ONLY if there is no specific sub_agent that can handle the sub_problem or the other sub_agents are already failed.
- If the sub_problem is correct and the failure is caused by the tools, don't repeat the same sub_agent if all of the tools in the sub_agent are already used and failed.

** Detailed agent selection guidelines: **
- **Visual-Agent**: Use for extracting information/text from images and answering simple questions about images. Choose when the sub_problem involves analyzing visual content, reading text from images, or understanding image-based information.
- **Browser_Extraction-Agent**: Use for extracting information or text directly from HTML web pages/URLs (landing pages). Best used after Search-Agent identifies relevant URLs. Choose when the sub_problem requires extracting content from a specific webpage or URL. Do NOT choose this agent for URLs that end with file types such as `.pdf`, `.jpg`, or other downloadable/attachment file extensions; use Download_File-Agent (for remote files) and then File_Extraction-Agent instead.
- **File_Extraction-Agent**: Use for extracting information/text from any type of files (PDF, DOCX, CSV, etc.). Only local files are supported. Works best with Download_File-Agent to download files first. Choose when the sub_problem involves reading or extracting data from file formats.
  - If the task is to **find/count which pages contain a search term/phrase** (e.g., “how many pages mention X”, “list page numbers containing Y”), make the sub_problem explicitly ask for: total page count, the list of page numbers with key phrase, and the count of pages with key phrase (avoid asking for full-document extraction).
  - If the agent fails to extract the information (invalid format or token limit), use General-Agent to deal with the problem.
- **Media-Agent**: Use for extracting information from media files like images, videos, or audio. Choose when the sub_problem involves processing multimedia content beyond simple image analysis.
- **Search-Agent**: Use for searching credible sources of information (Google search, Wikipedia, chemistry search, paper search, news search, etc.). Can only search for results, cannot extract information from search results. Choose when the sub_problem requires finding information sources or URLs, but not extracting content from them.
- **Download_File-Agent**: Use for downloading files from the web. Choose when the sub_problem requires obtaining a file from a URL before it can be processed. If the file has already been downloaded, don't try to download it again.
- **Puzzle-Agent**: Use for solving puzzles like mazes, Rubik's cubes, N-queens, etc. Cannot solve math problems, only puzzle problems. Choose when the sub_problem involves logic puzzles, pathfinding, or game-solving.
- **Mathematics-Agent**: Use for solving mathematics problems like algebra, geometry, calculus, or simple arithmetic calculations. Cannot solve puzzle problems, only pure mathematics problems. Choose when the sub_problem involves mathematical computation or problem-solving.
- **General-Agent**: Use as a fallback when no specific domain agent can handle the sub_problem. Only choose if no specialized agent is available or suitable for the task or the other sub_agents are already failed.
- If all the tools in the specialized sub_agent are already failed and no other sub_agent can handle the sub_problem, choose General-Agent to deal with the problem.
- If supporting_documents is a URL (or file path) that ends with ".pdf" (or other downloadable/attachment file extensions), you MUST NOT choose Browser_Extraction-Agent.
  - Choose Download_File-Agent if it is an HTTP/HTTPS URL (to download the PDF locally).
  - Choose File_Extraction-Agent if it is already a local PDF file path (to extract content from the PDF).

** supporting_documents selection: **
- Include necessary information like image, file, url, etc if exists and relevant to the sub_problem. If no supporting documents are available, set supporting_documents to None.
- The supporting documents must be a real link to a document that exists and is accessible without any other text or explanation.(Not a fake link or a link to a non-existent document.)

** Failure_source selection: **
- Choose exactly one:
  - "sub_problem": the goal/objective was wrong, underspecified, ambiguous, or aimed at the wrong target.
  - "sub_agent": the wrong domain agent was chosen for the sub_problem.
  - "tools": the goal and agent were reasonable, but the failure came from tool usage/arguments/retrieval/evidence handling.
- If all the tools in the specialized sub_agent are already failed and no other sub_agent can handle the sub_problem, choose General-Agent to deal with the problem.
** Anti-loop rule (critical): **
- If failure indicates repetition risk, follow these rules:
    - If failure_source is "tools": you MAY keep the same sub_problem, but refine it with extra constraints/details from the failure report to prevent the same tool mistake (do not name tools) or change the sub_agent to a more appropriate domain agent.
    - If failure_source is "sub_agent": you MAY keep the same sub_problem, but you MUST change sub_agent to a more appropriate domain agent. If all the tools in the specialized sub_agent are already failed and no other sub_agent can handle the sub_problem, choose General-Agent to deal with the problem.
    - If failure_source is "sub_problem": do NOT repeat the last attempted sub_problem verbatim. Change at least one of: target specificity or scope (or redirect the goal), and choose the appropriate sub_agent.

** Notes on common failure types: **
- If the failure is repetition/loop: choose a different approach (different sub_agent or narrower sub_problem) that changes what will happen next.
       """
    
    def make_next_step_prompt(self, query, sub_agent, global_memory, failure_report = None, image_path = None, file_path = None):
        input_prompt = f"""
Original user query: {query}
Available sub_agents: {sub_agent}
Memory of the system: {global_memory}
        """
        if image_path:
            input_prompt += f"\nImage: {image_path}"
        if file_path:
            input_prompt += f"\nFile: {file_path}"
        if failure_report:
            input_prompt += f"\nReason why there is still insufficient information to answer the original query: {failure_report}"
        return input_prompt

    def make_retry_prompt(self, query, sub_agent, global_memory, failure_report, image_path = None, file_path = None):
        input_prompt = f"""
Original user query: {query}
Available sub_agents: {sub_agent}
Memory of the system: {global_memory}
Failure report: {failure_report}
        """
        if image_path:
            input_prompt += f"\nImage: {image_path}"
        if file_path:
            input_prompt += f"\nFile: {file_path}"
        return input_prompt

    def reasoning_next_step(self, query, sub_agent, global_memory, state: bool = True, failure_report = None, image_path = None, file_path = None):
        if state:
            input_prompt = self.make_next_step_prompt(query, sub_agent, global_memory, failure_report, image_path, file_path)
            response = self.llm_engine.generate(query=input_prompt, response_format=NextStep, system_prompt=self.sys_prompt_next_step)
        else:
            input_prompt = self.make_retry_prompt(query, sub_agent, global_memory, failure_report, image_path, file_path)
            response = self.llm_engine.generate(query=input_prompt, response_format=RetryStep, system_prompt=self.sys_prompt_retry)
        return response
