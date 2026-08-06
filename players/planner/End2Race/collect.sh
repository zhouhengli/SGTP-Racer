#!/bin/bash

WORKERS=12
MAP_NAMES=("Spielberg" "Budapest" "Montreal" "Austin" "IMS" "Sepang" "Silverstone" "YasMarina" "Sochi" "warehouse_v2" "SaoPaulo" "Melbourne" "Norisring" "MexicoCity" "Zandvoort" "overtake_map" "Shanghai" "Spa" "Sakhir" "Catalunya" "warehouse_v0" "hangar" "torino" "mtl" "Monza")
SIM_DURATION=10
NUM_STARTPOINTS=30

# ============================================================================
# ETA helpers
# ============================================================================

format_seconds() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

TOTAL_JOBS=$((${#MAP_NAMES[@]} * NUM_STARTPOINTS))
START_TIME=$(date +%s)
STARTED_JOBS=0

# ============================================================================
# Main jobs
# ============================================================================

for map_name in "${MAP_NAMES[@]}"; do
    raceline_path="MapZoo/${map_name}/${map_name}_racing_line.csv"
    max_waypoints=$(tail -n +2 "$raceline_path" | wc -l)

    ego_idx_range=()
    for ((i=0; i<NUM_STARTPOINTS; i++)); do
        idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
        ego_idx_range+=($idx)
    done

    for ego_idx in "${ego_idx_range[@]}"; do

        cmd="python players/run_multi_method.py \
            --map_name $map_name \
            --num_agents 3 \
            --ego_idx $ego_idx \
            --sim_duration $SIM_DURATION"

        while [ "$(jobs -r | wc -l)" -ge "$WORKERS" ]; do
            sleep 0.1
        done

        STARTED_JOBS=$((STARTED_JOBS + 1))

        now=$(date +%s)
        elapsed=$((now - START_TIME))

        if [ "$STARTED_JOBS" -gt 0 ]; then
            avg_per_job=$((elapsed / STARTED_JOBS))
            remaining_jobs=$((TOTAL_JOBS - STARTED_JOBS))
            remain_seconds=$((avg_per_job * remaining_jobs))
        else
            remain_seconds=0
        fi

        elapsed_fmt=$(format_seconds "$elapsed")
        remain_fmt=$(format_seconds "$remain_seconds")

        echo "[$STARTED_JOBS/$TOTAL_JOBS] elapsed=$elapsed_fmt remain≈$remain_fmt"
        echo "$cmd"

        eval "$cmd" &
    done
done

wait

total_elapsed=$(( $(date +%s) - START_TIME ))
total_elapsed_fmt=$(format_seconds "$total_elapsed")

echo "All End2Race data collection jobs finished."
echo "Total elapsed: $total_elapsed_fmt"
