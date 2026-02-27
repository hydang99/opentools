# OpenTools Framework - Tools Directory

This directory contains all OpenTools tool modules. A tool is a self-contained component that exposes
metadata, an input schema, and a `run()` method so it can be discovered and executed consistently.

## Tool Inventory

Current tools are defined by their module directories and exported tool classes.  
The table below summarizes each tool and how it is evaluated:

- **Tool name**: clickable link to the tool’s folder.
- **Tool type**: coarse category, using one of:
  - `prompting_based` – mainly LLM reasoning / prompting.
  - `api_based` – primarily calls an external HTTP / API service.
  - `programming_based` – mostly pure Python / local file processing.
- **Evaluated?**:
  - `✅` = this tool folder currently contains a `test_result.json` file (it has been run against the shared test file).
  - `☐` = no `test_result.json` present yet (not evaluated with the shared test file).
- **Test suite key**: name used in `test_file/data.json`.
- **Evaluation metrics**: typical `search_type` used (`exact_match`, `similarity_eval`, `search_pattern`).
- **Current accuracy**: for evaluated tools, this column shows the **average** of `Final_Accuracy.run_1 / run_2 / run_3` (in %). Cells marked `–` mean no `test_result.json` is available yet.

| Tool name (folder) | Short description | Tool type | Evaluated? | Test suite key (`test_file/data.json`) | Evaluation metrics | Current accuracy |
|--------------------|-------------------|-----------|------------|----------------------------------------|--------------------|------------------|
| [`advanced_object_detector`](./advanced_object_detector/) | Detects and localizes objects in images. | prompting_based | ☐ | `advanced_object_detector` | similarity_eval | – |
| [`advanced_text_detector`](./advanced_text_detector/) | Detects and extracts text from images (OCR). | prompting_based | ✅ | `advanced_text_detector` | exact_match / similarity_eval | 83.54 |
| [`archived`](./archived/) | Placeholder for deprecated or archived tools. | local_processing | ☐ | `archived` | – | – |
| [`arxiv_paper_search`](./arxiv_paper_search/) | Searches and retrieves papers from arXiv. | api_based | ✅ | `arxiv_paper_search` | search_pattern / similarity_eval | 97.78 |
| [`audio_processing`](./audio_processing/) | Processes and analyzes audio files. | local_processing | ✅ | `audio_processing` | exact_match / similarity_eval | 90.0 |
| [`board_title_solver`](./board_title_solver/) | Solves board or title-based puzzles. | prompting_based | ✅ | `board_title_solver` | exact_match | 100.0 |
| [`browser_interaction`](./browser_interaction/) | Controls and automates browser actions. | api_based | ✅ | `browser_interaction` | task-specific | 94.93 |
| [`calculator`](./calculator/) | Performs arithmetic and calculations. | local_processing | ✅ | `calculator` | exact_match | 99.0 |
| [`calendar_calculation`](./calendar_calculation/) | Performs date and calendar computations. | local_processing | ✅ | `calendar_calculation` | exact_match | 100.0 |
| [`chemistry_search`](./chemistry_search/) | Searches chemistry-related databases or information. | api_based | ✅ | `chemistry_search` | search_pattern / similarity_eval | 88.89 |
| [`code_generate_execute`](./code_generate_execute/) | Generates and executes code. | prompting_based | ✅ | `code_generate_execute` | task-specific | 55.90 |
| [`colour_hue_solver`](./colour_hue_solver/) | Solves colour or hue-based puzzles. | prompting_based | ✅ | `colour_hue_solver` | exact_match | 100.0 |
| [`csv_extraction`](./csv_extraction/) | Extracts data from CSV files. | local_processing | ✅ | `csv_extraction` | exact_match | 100.0 |
| [`doc_extraction`](./doc_extraction/) | Extracts text and structure from documents. | local_processing | ✅ | `doc_extraction` | exact_match / similarity_eval | 100.0 |
| [`download_file`](./download_file/) | Downloads files from URLs. | api_based | ✅ | `download_file` | exact_match | 100.0 |
| [`generalist_solution_generator`](./generalist_solution_generator/) | Generates solutions for general tasks using an LLM. | prompting_based | ☐ | `generalist_solution_generator` | similarity_eval | – |
| [`google_search_octotools`](./google_search_octotools/) | Performs web search via Google (Octotools integration). | api_based | ☐ | `google_search_octotools` | search_pattern | – |
| [`math_solver`](./math_solver/) | Solves mathematical problems (optionally with images). | prompting_based | ✅ | `math_solver` | exact_match | 67.39 |
| [`maze_solving`](./maze_solving/) | Solves maze puzzles. | local_processing | ✅ | `maze_solving` | exact_match | 100.0 |
| [`n_queens_solving`](./n_queens_solving/) | Solves the N-queens puzzle. | local_processing | ✅ | `n_queens_solving` | exact_match | 100.0 |
| [`nature_news_fetcher`](./nature_news_fetcher/) | Fetches news or content from Nature. | api_based | ☐ | `nature_news_fetcher` | search_pattern | – |
| [`pdf_extraction`](./pdf_extraction/) | Extracts text and content from PDFs. | local_processing | ✅ | `pdf_extraction` | exact_match / similarity_eval | 100.0 |
| [`plain_text_extraction`](./plain_text_extraction/) | Extracts plain text from files. | local_processing | ✅ | `plain_text_extraction` | exact_match / similarity_eval | 90.0 |
| [`pptx_extraction`](./pptx_extraction/) | Extracts content from PowerPoint files. | local_processing | ✅ | `pptx_extraction` | exact_match / similarity_eval | 100.0 |
| [`pubmed_search`](./pubmed_search/) | Searches the PubMed database. | api_based | ✅ | `pubmed_search` | search_pattern / similarity_eval | 53.72 |
| [`relevant_patch_zoomer`](./relevant_patch_zoomer/) | Finds or zooms into relevant image patches. | prompting_based | ☐ | `relevant_patch_zoomer` | similarity_eval | – |
| [`rubik_cube_solver`](./rubik_cube_solver/) | Solves Rubik's cube configurations. | local_processing | ✅ | `rubik_cube_solver` | exact_match | 0.0 |
| [`search_engine`](./search_engine/) | Performs general web search. | api_based | ✅ | `search_engine` | search_pattern / similarity_eval | 70.81 |
| [`simple_arxiv_paper_search`](./simple_arxiv_paper_search/) | Simplified arXiv paper search. | api_based | ☐ | `simple_arxiv_paper_search` | search_pattern / similarity_eval | – |
| [`target_solver`](./target_solver/) | Solves target-based puzzles or tasks. | prompting_based | ✅ | `target_solver` | exact_match | 99.36 |
| [`text_detector`](./text_detector/) | Detects and extracts text from images. | prompting_based | ✅ | `text_detector` | exact_match / similarity_eval | 71.24 |
| [`url_text_extractor`](./url_text_extractor/) | Extracts text from a URL's web page. | api_based | ✅ | `url_text_extractor` | search_pattern / similarity_eval | 90.91 |
| [`url_text_extractor_octotools`](./url_text_extractor_octotools/) | URL text extraction (Octotools integration). | api_based | ☐ | `url_text_extractor_octotools` | search_pattern / similarity_eval | – |
| [`video_processing`](./video_processing/) | Processes and analyzes video files. | local_processing | ☐ | `video_processing` | exact_match / similarity_eval | – |
| [`visual_ai`](./visual_ai/) | Analyzes images and answers visual questions (multimodal). | prompting_based | ✅ | `visual_ai` | similarity_eval | 78.04 |
| [`wiki_search`](./wiki_search/) | Searches Wikipedia. | api_based | ✅ | `wiki_search` | search_pattern / similarity_eval | 70.76 |
| [`wikipedia_knowledge_searcher_octotools`](./wikipedia_knowledge_searcher_octotools/) | Wikipedia search (Octotools integration). | api_based | ☐ | `wikipedia_knowledge_searcher_octotools` | search_pattern / similarity_eval | – |
| [`wolfram_math`](./wolfram_math/) | Mathematical computation via Wolfram. | api_based | ✅ | `wolfram_math` | exact_match | 64.99 |
| [`woodslide_solver`](./woodslide_solver/) | Solves woodslide puzzles. | local_processing | ✅ | `woodslide_solver` | exact_match | 100.0 |
| [`xlsxe_extraction`](./xlsxe_extraction/) | Extracts data from Excel files. | local_processing | ✅ | `xlsxe_extraction` | exact_match | 100.0 |
| [`yahoo_finance`](./yahoo_finance/) | Fetches financial data from Yahoo Finance. | api_based | ☐ | `yahoo_finance` | exact_match | – |
| [`youtube`](./youtube/) | Fetches or searches YouTube content. | api_based | ☐ | `youtube` | search_pattern / similarity_eval | – |

## Tool Architecture

Each tool lives in its own package directory:

```
src/opentools/tools/your_tool_name/
  __init__.py
  tool.py
```

- `tool.py` contains the `BaseTool` subclass and core logic.
- `__init__.py` imports the tool class and sets `__all__` to expose it for discovery.
- Tool packages are auto-discovered by `ToolRegistry` when `opentools.tools` is imported
  or when `load_all_tools()` is called.

## BaseTool Required Parameters

`BaseTool.__init__` requires these parameters:

- `name`: Unique tool name.
- `description`: Full description including capabilities and examples.
- `category`: Category string used for grouping.
- `tags`: List of search tags.
- `parameters`: OpenAI-compatible JSON schema for tool inputs.
- `agent_type`: Intended agent type (for routing and metadata).
- `demo_commands`: Example command(s) and descriptions.

Minimal signature shape:

```python
BaseTool(
    name,
    description,
    category,
    tags,
    parameters,
    agent_type,
    demo_commands,
    # optional fields below...
)
```

## BaseTool Optional Parameters

Common optional parameters:

- `limitation`: Known limitations.
- `type`: Tool type, default `"function"`.
- `strict`: Enforce schema validation, default `True`.
- `accuracy`: Accuracy metrics (often loaded from `test_result.json`).
- `model_string`: LLM model identifier if the tool uses an engine.
- `required_api_keys`: List of API key names (e.g., `["WEATHER_API_KEY"]`).
- `is_multimodal`: Whether the tool is multimodal.
- `llm_engine`: Injected engine instance.
- `require_llm_engine`: If `True`, BaseTool will create an engine when not provided.

## BaseTool Utility Methods

Useful helpers available on every tool:

- `get_metadata()`: Returns the tool metadata dictionary used by the registry.
- `set_custom_output_dir(path)`: Overrides the output directory for the tool.
- `get_api_key(name)`: Reads an API key from the OpenTools config.
- `require_api_key(name)`: Reads an API key and raises if missing.
- `check_required_api_keys()`: Validates all keys in `required_api_keys`.
- `embed_tool()`: Generates and saves tool embeddings when an LLM engine is available.
- `find_accuracy(path)`: Reads accuracy metrics from a JSON result file.
- `test(...)`: Runs the built-in test runner against `test_file/data.json`.

## How to Run a Tool


Direct instantiation is also supported:

```python
from opentools.tools.calculator import Calculator_Tool

tool = Calculator_Tool()
print(tool.run(operation="add", values=[1, 2, 3]))
```

## How to Create a Tool

### 1) Create the Module

```bash
mkdir src/opentools/tools/your_tool_name

touch src/opentools/tools/your_tool_name/__init__.py

touch src/opentools/tools/your_tool_name/tool.py
```

### 2) Implement the Tool Class

```python
import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opentools.core.base import BaseTool

class Your_Tool_Name(BaseTool):
    """Brief description of what your tool does."""

    def __init__(self):
        super().__init__(
            name="Your_Tool_Name",
            description="Comprehensive description, synonyms, and examples.",
            category="your_category",
            tags=["relevant", "search", "tags"],
            parameters={
                "type": "object",
                "properties": {
                    "input_param": {
                        "type": "string",
                        "description": "Description of the parameter"
                    }
                },
                "required": ["input_param"],
                "additionalProperties": False,
            },
            agent_type="specialized",
            demo_commands={
                "command": "response = tool.run(input_param='example')",
                "description": "Example usage",
            },
            limitation="Known limitations and constraints",
            strict=True,
            required_api_keys=["YOUR_API_KEY"],
        )

    def run(self, input_param: str, **kwargs) -> str:
        if not input_param or not input_param.strip():
            raise ValueError("input_param cannot be empty")
        return input_param
```

### 3) Export in `__init__.py`

```python
"""Your tool module."""
from .tool import Your_Tool_Name

__all__ = ["Your_Tool_Name"]
```

## Categories

Choose the category that best matches your tool:

- `mathematics`
- `web_search`
- `web_automation`
- `document_processing`
- `image_processing`
- `audio_processing`
- `video_processing`
- `research`
- `problem_solving`
- `data_extraction`
- `utilities`
- `development`

## Tags

Tags improve discovery and routing. Keep them short and descriptive:

```python
tags=[
    "primary_function",
    "domain",
    "capability",
    "use_case",
]
```

## API Key Management

- Declare required keys using `required_api_keys` in `BaseTool.__init__`.


## Testing

The built-in test helper reads test cases from `src/opentools/tools/test_file/data.json` and
writes results to `temp_result.json` under the tool directory (`tools/<file_location>/temp_result.json`).

```python
tool.test(
    tool_test=<name_of_test_in_data.json>,
    file_location=<path_to_tool_relative_to_src/opentools/tools>,
    result_parameter=<field_with_result>,
    search_type=<evaluation_mode>,
)
```

`search_type` supports `exact_match`, `similarity_eval`, and `search_pattern`.

---

## Contributing test cases

We welcome test cases from outside contributors. Tests are stored in a shared **test file folder** and evaluated with the same runner and metrics used by the project.

### Where tests live

- **Directory:** `src/opentools/tools/test_file/`
- **Main data file:** `test_file/data.json`
- **Assets:** Put all test inputs under `test_file/` in the subfolder that matches the data type (see below). Reference them in `data.json` by paths **relative to** `test_file/`.

### File and folder classification

Keep test assets organized by type. Use these subfolders under `test_file/`; if no folder fits your data type, create a new one with a clear name (e.g. `spreadsheets/`, `archives/`).

| Data type | Subfolder | Example paths in `data.json` |
|-----------|-----------|------------------------------|
| **Images** (jpg, png, gif, webp, etc.) | `images/` or `ocr/` (for OCR tests) | `"image_path": "images/sample.jpg"`, `"image": "ocr/2_0.jpg"` |
| **Audio** (wav, mp3, flac, etc.) | `audio/` | `"file_path": "audio/sample.wav"` |
| **Video** (mp4, webm, etc.) | `video/` | `"file_path": "video/sample.mp4"` |
| **CSV / TSV / spreadsheets** | `csvs/` or `tables/` | `"file_path": "csvs/data.csv"` |
| **Documents** (PDF, DOC, DOCX) | `documents/` or `docs/` | `"file_path": "documents/sample.pdf"` |
| **Presentations** (PPTX) | `presentations/` or `pptx/` | `"file_path": "presentations/slide.pptx"` |
| **Plain text** (.txt, .md, .rst) | `files/` or `text/` | `"file_path": "files/readme.txt"` |
| **Web** (HTML, CSS, JS) | `files/web/` | `"file_path": "files/web/page1.html"` |
| **Structured data** (JSON, XML, YAML) | `files/data/` or `files/cfg/` | `"file_path": "files/data/items.xml"` |
| **Config / settings** (.ini, .properties, .conf) | `files/settings/` or `files/cfg/` | `"file_path": "files/settings/app.ini"` |
| **Other or mixed** | `files/` | Use for anything that doesn’t fit above. |

- Paths in `data.json` are always **relative to** `test_file/` (e.g. `images/photo.png`, not `test_file/images/photo.png`).
- If you need a folder that doesn’t exist yet (e.g. `audio/`, `video/`, `documents/`), create it under `test_file/` and add your assets there.

### How to add test cases

1. **Open** `src/opentools/tools/test_file/data.json`.
2. **Choose or create a test suite.** The JSON is keyed by suite name (e.g. `"text_detector"`, `"arxiv_paper_search"`). Each suite is a list of test cases.
3. **Add a new case** to the appropriate list. Each case is one object with:
   - **`id`**: Unique integer for the case (optional but useful).
   - **`answer`**: Expected value used to compute accuracy (required).
   - **`category`**: Optional label (e.g. tool or task name).
   - **Input parameters**: Any fields the tool’s `run()` expects (e.g. `image_path`, `file_path`, `query`, `url`). Use paths **relative to** `test_file/` and follow the [file and folder classification](#file-and-folder-classification) above (e.g. images → `images/` or `ocr/`, audio → `audio/`, CSV → `csvs/`). The test runner resolves these to absolute paths.
4. **Save** `data.json`. Place any new assets in the right subfolder under `test_file/` (create the folder if it doesn’t exist).

Example: adding one case to the `text_detector` suite:

```json
{
  "text_detector": [
    {
      "id": 0,
      "image_path": "ocr/my_image.jpg",
      "answer": "Expected extracted text here",
      "category": "text_detector"
    }
  ]
}
```

To add a **new** test suite, add a new top-level key and a list of cases with the same shape.

### How to run the tests

From the tool’s `tool.py` (e.g. in `if __name__ == "__main__"`), call:

```python
tool.test(
    tool_test="<suite_name>",           # key in data.json (e.g. "text_detector")
    file_location="<tool_folder>",      # e.g. "text_detector" or "visual_ai"
    result_parameter="<result_key>",    # key in run() output holding the answer (e.g. "result")
    search_type="<metric>",             # one of: exact_match, similarity_eval, search_pattern
    count_token=False,                  # set True to record token usage
)
```

Run the tool as a script (e.g. `python tool.py`) so this `test()` call runs. Results are written to `tools/<file_location>/temp_result.json`.

### Which metrics are used

Evaluation is controlled by **`search_type`**:

| `search_type`      | Description |
|--------------------|-------------|
| **`exact_match`**  | Tool output (from `result_parameter`) must equal the case’s `answer` exactly. Best for deterministic outputs (e.g. numbers, fixed strings). |
| **`similarity_eval`** | Semantic similarity between tool output and `answer` (cosine similarity of embeddings). Best for paraphrased or flexible text. |
| **`search_pattern`**  | Tool output is correct if the string `answer` appears in the output (case-insensitive). Best when the answer is a substring or keyword. |

Each test case is run multiple times (e.g. 3 runs); accuracy is aggregated and written to `temp_result.json` under `Final_Accuracy` (e.g. `run_1`, `run_2`, `run_3` as percentages).