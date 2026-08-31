#!/bin/bash

# Score the folders jobs/fid_10k_generate.sh produced. Same args:
#   bash jobs/tri_quality_eval.sh nudity church          (default: all three)
JOBS=("fid" "clip")
CONCEPTS=(nudity VanGogh church)
[ $# -gt 0 ] && CONCEPTS=("$@")

CKPT_DIR="results/results_with_AEGIS/AEGIS/models"
NUMS=("999")

DIRS=()
for c in "${CONCEPTS[@]}"; do DIRS+=("$CKPT_DIR/Diffusers-UNet-full-$c-epoch"); done

GPU_START=0
GPU_END=$(( $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) - 1 ))

JOB_NUM=1

# Helper function to check GPU availability
check_gpu() {
    GPU_ID=$1
    # Check if the number of processes is less than 2
    [[ $(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader | wc -l) -lt $JOB_NUM ]]
}

# Main loop
for job in "${JOBS[@]}"; do
    for dir in "${DIRS[@]}"; do 
        for num in "${NUMS[@]}"; do 
            # Find an available GPU
            GPU=-1
            while [[ $GPU -lt $GPU_START ]]; do
                for (( i=$GPU_START; i<=$GPU_END; i++ )); do
                    if check_gpu $i; then
                        GPU=$i
                        break
                    fi
                done
                # If no GPU with less than two processes found, sleep for a while before checking again
                if [[ $GPU -lt $GPU_START ]]; then
                    sleep 15
                fi
            done

            if [[ $job == "fid" ]]; then
                suffix="visualizations_fid_10k"
            elif [[ $job == "clip" ]]; then
                suffix="visualizations_fid_10k"
            fi


            # Run the job on the available GPU
            CUDA_VISIBLE_DEVICES=$GPU python train-scripts/img_retain_eval.py \
                --gen_imgs_path "${dir}_${num}_${suffix}/SD-v1-4" \
                --job $job &

            # Sleep for a bit to make sure the job starts
            sleep 15
        done
    done
done
# Wait for all background jobs to complete
wait
echo "All jobs completed."
