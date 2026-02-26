#!/bin/bash

############ [1] Batching Run ############
PROJECT_DIR="../"

############
LABEL="gpt-4o-mini/ReAct_OpenTools_Toolset"
RUN_ID="run_1"
THREADS=16
TASK="scifibench"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL/$RUN_ID"
OUT_DIR="$TASK/results/$LABEL/$RUN_ID"
ERROR_JSON_DIR="$TASK/error_json/$LABEL/$RUN_ID"    
CACHE_DIR="$TASK/cache"
LLM="gpt-4o-mini"


ENABLED_TOOLS="Visual_AI_Tool,Advanced_Text_Detector_Tool,Generalist_Solution_Generator_Tool,Wiki_Search_Tool,Arxiv_Paper_Search_Tool"
AGENTS="react"
############

cd $PROJECT_DIR
mkdir -p $LOG_DIR
mkdir -p $ERROR_JSON_DIR
mkdir -p $OUT_DIR
# Define the array of specific indices
indices=($(seq 0 299))

# Skip indices if the output file already exists
new_indices=()
for i in "${indices[@]}"; do
    if [ ! -f "$OUT_DIR/output_$i.json" ]; then
        new_indices+=($i)
    else
        echo "Output file already exists: $OUT_DIR/output_$i.json"
    fi
done
indices=("${new_indices[@]}")
echo "Final indices: ${indices[@]}"

# Check if indices array is empty
if [ ${#indices[@]} -eq 0 ]; then
    echo "All tasks completed."
else
    # Function to run the task for a single index
    run_task() {
        local i=$1
        echo "Running task for index $i"
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
        --error_json_dir $ERROR_JSON_DIR \
        2>&1 | tee $LOG_DIR/$i.log
        echo "Completed task for index $i"
        echo "------------------------"
    }


    # Export the function and variables so they can be used by parallel
    export -f run_task
    export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS AGENTS ERROR_JSON_DIR

    # Run the tasks in parallel using GNU Parallel
    echo "Starting parallel execution (bash-only)…"
    for i in "${indices[@]}"; do
        # throttle to $THREADS concurrent jobs
        while [ "$(jobs -rp | wc -l)" -ge "$THREADS" ]; do
        sleep 0.5
        done
        run_task "$i" &  
    done
    echo "Waiting for all tasks to finish…"
    wait
fi


############ [2] Calculate Scores ############


RESPONSE_TYPE="direct_output"
echo "Calculating scores"
python $TASK/calculate_score.py \
--data_file $DATA_FILE \
--result_dir $OUT_DIR \
--response_type $RESPONSE_TYPE \
--output_file "final_results.json" \
| tee "$OUT_DIR/final_results.log"

