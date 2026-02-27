# OpenTools Benchmark

This directory contains benchmark datasets and a multi-agent runner for evaluating OpenTools agents on standardized tasks. You can run a single problem or batch runs via the command line or shell scripts, then score results with task-specific scripts. We inherits some of calculate_score from OctoTools paper for fair comparisons and sample scripts in each benchmark folder. 

---

## Directory structure

```
src/opentools/Benchmark/
  solve.py              # Main entry: runs one or more agents on a dataset
  utils.py               # Helpers: ResultAnalyzer (time, steps, tool usage)
  README.md             # This file

  <benchmark_name>/      # One folder per benchmark (e.g. clevr-math, vqav2)
    data/
      data.json          # Required: problem set (query, answer, image, etc.)
    results/             # Output JSONs (written by solve.py)
    logs/                # Per-run logs (if using .sh scripts)
    error_json/          # Error dumps for failed runs
    cache/               # Solver cache (optional)
    calculate_score.py   # Optional: task-specific scoring script
    *.sh                 # Optional: shell scripts for batch runs
```

**Existing benchmarks** (each has a subfolder under `Benchmark/`):  
`algopuzzlevqa`, `clevr-math`, `gaia-text`, `gameof24`, `gpqa`, `hallusion-vd`, `mathvista`, `medqa`, `mmlu-pro`, `omni-math`, `pathvqa`, `puzzlevqa`, `scifibench`, `slake`, `vqav2`, and others.

---

## Dataset format

`solve.py` loads a JSON file (typically `<benchmark_name>/data/data.json`) that is a **list of problem objects**. Each object can have:

| Field       | Required | Description |
|------------|----------|-------------|
| `query` or `question` | Yes  | The problem text. |
| `answer`   | No (but needed for scoring) | Ground-truth answer. |
| `image`    | No  | Image path **relative to the data file’s directory** (e.g. `images/0.png`). Resolved to absolute path by `solve.py`. |
| `unit`     | No  | If present, appended to the query as “Please answer in the unit of {unit}”. |
| `pid`      | No  | Problem ID; used in outputs and scoring. |
| `metadata` | No  | Extra fields; passed through to results. |

Example minimal record:

```json
{
  "query": "What is 2 + 2?",
  "answer": "4",
  "pid": "0"
}
```

With image:

```json
{
  "query": "How many objects are in this image?",
  "answer": "3",
  "image": "images/sample.png",
  "pid": "0"
}
```

---

## How to run

### From the command line

Run from the **`Benchmark`** directory (so that `solve.py` is in the current directory), or pass paths relative to your cwd.

**Single problem (e.g. index 0):**

```bash
cd src/opentools/Benchmark

python solve.py \
  --task clevr-math \
  --data_file clevr-math/data/data.json \
  --agents opentools,react,zero_shot \
  --llm_engine_name gpt-4o-mini \
  --index 0
```

**Single problem with tool restrictions and FAISS retrieval:**

```bash
python solve.py \
  --task clevr-math \
  --data_file clevr-math/data/data.json \
  --agents opentools,react \
  --llm_engine_name gpt-4o-mini \
  --enabled_tools Calculator_Tool,Calendar_Calculation_Tool \
  --enable_faiss_retrieval True
```

### Using the shell scripts (batch runs)

Each benchmark folder often contains `.sh` scripts (e.g. `react_gpt4_opentools_toolset.sh`). They:

1. **cd** into the parent directory (so the current directory is `Benchmark/`).
2. Run **solve.py** for many indices (e.g. 0–299), optionally in parallel.
3. Call **calculate_score.py** for that benchmark to produce final scores.

Run a script **from inside the benchmark folder**:

```bash
cd src/opentools/Benchmark/clevr-math
bash react_gpt4_opentools_toolset.sh
```

The script uses variables at the top to set task, data file, LLM, agents, tools, and paths; see below for how to change them.

---

## Modifying the bash script (LLM, agents, tools, and other parameters)

Scripts like `react_gpt4_opentools_toolset.sh` define variables at the top, then pass them into `solve.py`. You can change behavior by editing these.

### 1. Where to edit

Open the `.sh` file. Near the top you’ll see a block like:

```bash
PROJECT_DIR="../"
LABEL="gpt-4o-mini/ReAct_OpenTools_Toolset"
RUN_ID="run_1"
THREADS=16
TASK="clevr-math"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL/$RUN_ID"
OUT_DIR="$TASK/results/$LABEL/$RUN_ID"
ERROR_JSON_DIR="$TASK/error_json/$LABEL/$RUN_ID"
CACHE_DIR="$TASK/cache"
LLM="gpt-4o-mini"
ENABLED_TOOLS="Relevant_Patch_Zoomer_Tool,Visual_AI_Tool,..."
AGENTS="react"
```

And later, `python solve.py` is called with options like `--llm_engine_name $LLM`, `--agents "$AGENTS"`, etc.

### 2. What each variable does (and what to change)

| Variable | Purpose | Example change |
|----------|--------|-----------------|
| **`LLM`** | Model name passed to `--llm_engine_name`. | `LLM="gpt-4o"` or `LLM="gpt-5-mini"` to switch model. |
| **`AGENTS`** | Comma-separated agents for `--agents`. | `AGENTS="react,opentools"` or `AGENTS="zero_shot"`. |
| **`ENABLED_TOOLS`** | Comma-separated tool list for `--enabled_tools`. Leave empty in the script to use all tools. | `ENABLED_TOOLS="Calculator_Tool,Visual_AI_Tool"` to restrict tools. |
| **`TASK`** | Benchmark name (used in metadata and paths). | Set to your benchmark folder name, e.g. `TASK="my-benchmark"`. |
| **`DATA_FILE`** | Path to `data.json`, usually `$TASK/data/data.json`. | Change only if your data file is elsewhere. |
| **`LABEL`** | Used in `LOG_DIR`, `OUT_DIR`, `ERROR_JSON_DIR` to separate runs (e.g. by model and config). | e.g. `LABEL="gpt-4o/react_only"` to avoid overwriting other runs. |
| **`RUN_ID`** | Further subfolder for this run (e.g. `run_1`, `run_2`). | Change to keep multiple runs under the same `LABEL`. |
| **`THREADS`** | Max concurrent jobs in the bash loop. | Increase for more parallelism (e.g. `THREADS=32`). |

### 3. Adding or changing solve.py arguments in the script

Inside the script, `run_task()` runs something like:

```bash
python solve.py \
  --index $i \
  --task $TASK \
  --data_file $DATA_FILE \
  --llm_engine_name $LLM \
  --root_cache_dir $CACHE_DIR \
  --output_json_dir $OUT_DIR \
  --output_types direct \
  --enabled_tools "$ENABLED_TOOLS" \
  --agents "$AGENTS" \
  --max_time 750 \
  --max_steps 10 \
  --max_tokens 16000 \
  --error_json_dir $ERROR_JSON_DIR
```

To change behavior, add or edit flags here. Useful options:

- **`--enable_faiss_retrieval True`** — Use FAISS tool retrieval (add the flag in the `python solve.py` call).
- **`--max_time 1200`** — Allow more seconds per problem.
- **`--max_steps 15`** — Allow more reasoning/tool steps.
- **`--max_tokens 8000`** — Cap tokens per LLM call.
- **`--output_types base,final,direct,full`** — Request multiple output types.
- **`--verbose True`** or **`False`** — Control logging.

So: **LLM** → set `LLM` and ensure `--llm_engine_name $LLM` is present. **Agents** → set `AGENTS`. **Tools** → set `ENABLED_TOOLS` and `--enabled_tools "$ENABLED_TOOLS"`. **Limits** → edit `--max_time`, `--max_steps`, `--max_tokens` in the `python solve.py` call.

### 4. Indices (which problems to run)

Scripts often build a list of indices, e.g. `indices=($(seq 0 299))`. To run a subset:

- Change the range: e.g. `indices=($(seq 0 99))` for the first 100 problems.

The script may skip indices that already have an output file in `OUT_DIR` so you can resume.

### 5. Scoring step at the end

After the batch run, the script usually calls that benchmark’s `calculate_score.py`, e.g.:

```bash
python $TASK/calculate_score.py \
  --data_file $DATA_FILE \
  --result_dir $OUT_DIR \
  --response_type $RESPONSE_TYPE \
  --output_file "final_results.json"
```

`RESPONSE_TYPE` (e.g. `direct_output` or `final_answer`) must match the field name your agents write and what `calculate_score.py` expects. Set it at the top of the script if you add a variable for it.

---

## Full list of solve.py arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `minitoolbench` | Task name (used in metadata). |
| `--data_file` | `data/data.json` | Path to benchmark JSON. |
| `--task_description` | (empty) | String prepended to every query. |
| `--agents` | `opentools,octotools,react,chain_of_thought,zero_shot` | Comma-separated agent names. |
| `--llm_engine_name` | `gpt-4o-mini` | Model name (e.g. `gpt-4o`, `gpt-5-mini`). |
| `--index` | `0` | Single problem index to run (optional; if set, only this index is run). |
| `--output_types` | `base,final,direct,full` | Comma-separated: which output fields to request. |
| `--enabled_tools` | (empty = all) | Comma-separated tool class names. |
| `--enable_faiss_retrieval` | `False` | Use FAISS to retrieve relevant tools. |
| `--max_steps` | `10` | Max steps per problem. |
| `--max_time` | `300` | Max seconds per problem. |
| `--max_tokens` | `4000` | Max tokens per LLM call. |
| `--output_json_dir` | `results` | Directory for output JSONs. |
| `--error_json_dir` | `error_json` | Directory for error JSONs. |
| `--root_cache_dir` | `solver_cache` | Solver cache directory. |
| `--verbose` | `True` | Verbose logging. |
| `--list_agents` | (flag) | List registered agents and exit. |

---

## Outputs

- **Single agent:** outputs are written as `output_<index>.json` (or `output_<index>_<agent>.json` when multiple agents are used) under `--output_json_dir`.
- Each file contains the problem (query, image, answer, etc.) plus agent output fields such as `direct_output`, `final_output`, `base_output`, `full_answer`, and metadata (task, agents, llm_engine, steps, time, etc.).
- Failed runs can be logged under `--error_json_dir` (script-dependent).

---

## Result analysis (utils.py)

In Python you can use:

- **`ResultAnalyzer.calculate_time_steps(log_dir)`** — Returns average time and steps (from logs or from result JSONs if logs are missing).
- **`ResultAnalyzer.calculate_tool_usage(result_dir, return_counts=...)`** — Returns tool usage statistics from result JSONs.

---

## How to add a new benchmark

To run a new dataset and (optionally) score it with a script:

### 1. Create the benchmark folder and data file

```text
src/opentools/Benchmark/my_benchmark/
  data/
    data.json    # List of problems (see "Dataset format" above)
```

Put your problems in `data.json` with at least `query` (or `question`) and, for scoring, `answer`. Use `image` with paths relative to the `data/` directory if you have images.

### 2. Run with solve.py (no script)

From `Benchmark/`:

```bash
python solve.py \
  --task my_benchmark \
  --data_file my_benchmark/data/data.json \
  --agents react \
  --llm_engine_name gpt-4o-mini \
  --index 0
```

Adjust `--agents`, `--llm_engine_name`, `--enabled_tools`, etc. as needed. For a range of indices you’ll need a loop or a script (next step).

### 3. Add a shell script for batch runs (optional)

Copy an existing script from another benchmark (e.g. `clevr-math/react_gpt4_opentools_toolset.sh`) into your folder:

```bash
cp src/opentools/Benchmark/clevr-math/react_gpt4_opentools_toolset.sh \
   src/opentools/Benchmark/my_benchmark/
```

Edit the script:

- Set **`TASK="my_benchmark"`** so `DATA_FILE` becomes `my_benchmark/data/data.json`.
- Set **`LLM`**, **`AGENTS`**, **`ENABLED_TOOLS`** (and optionally **`LABEL`**, **`RUN_ID`**, **`THREADS`**).
- Set **`indices`** to your problem indices, e.g. `indices=($(seq 0 99))` for 100 problems.
- Ensure the **`python solve.py ...`** call includes all flags you need (e.g. `--enable_faiss_retrieval True`).

Run from inside your benchmark folder:

```bash
cd src/opentools/Benchmark/my_benchmark
bash react_gpt4_opentools_toolset.sh
```

Outputs go to `my_benchmark/results/...` and logs to `my_benchmark/logs/...` (paths are set by the script).

### 4. Add scoring (optional)

If you want automatic scoring after the batch:

1. **Add `calculate_score.py`** in your benchmark folder (e.g. by copying and adapting one from `clevr-math` or `vqav2`). It should:
   - Load `--data_file` (benchmark data) and `--result_dir` (output JSONs).
   - Match results to problems (e.g. by `pid` or by filename `output_<index>.json`).
   - Read the agent response from the field you use (e.g. `direct_output`, `final_answer`).
   - Compare to `answer` and compute accuracy (or your metric).
   - Write a summary (e.g. `final_results.json`) and optionally print stats.

2. **In your .sh script**, after the batch run, call:

   ```bash
   RESPONSE_TYPE="direct_output"   # or whatever field your agents produce
   python $TASK/calculate_score.py \
     --data_file $DATA_FILE \
     --result_dir $OUT_DIR \
     --response_type $RESPONSE_TYPE \
     --output_file "final_results.json" \
     | tee "$OUT_DIR/final_results.log"
   ```

   Use the same `OUT_DIR` and `DATA_FILE` as in the batch run, and set `RESPONSE_TYPE` to the key you use in your result JSONs (e.g. `direct_output`).

Once this is in place, your new benchmark is runnable via the same pattern as the existing ones: configure variables and `solve.py` args in the script, run the script from the benchmark folder, then score with `calculate_score.py`.
