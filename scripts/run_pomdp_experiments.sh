#!/bin/bash
# POMDP experiments batch runner
# Usage: ./scripts/run_pomdp_experiments.sh [env_name] [num_seeds]

set -e

ENV_NAME=${1:-"Pendulum-V-v0"}
NUM_SEEDS=${2:-5}

# Configuration
MODELS=("rnn" "snn")
SNN_TYPES=("LIF" "RecurrentLIF" "GRSNwoTAP")
ENCODER="gru"
ALGO="sac"

echo "Running POMDP experiments on ${ENV_NAME} with ${NUM_SEEDS} seeds"
echo "================================================"

for ((seed=0; seed<NUM_SEEDS; seed++)); do
    echo ""
    echo "===== Seed ${seed} ====="

    # RNN baseline
    echo "Training RNN (${ENCODER})..."
    python experiments/train.py \
        --env ${ENV_NAME} \
        --model_type rnn \
        --encoder ${ENCODER} \
        --algo ${ALGO} \
        --seed ${seed}

    # SNN variants
    for snn_type in "${SNN_TYPES[@]}"; do
        echo "Training SNN (${snn_type})..."
        python experiments/train.py \
            --env ${ENV_NAME} \
            --model_type snn \
            --snn_type ${snn_type} \
            --algo ${ALGO} \
            --seed ${seed}
    done
done

echo ""
echo "All experiments completed!"
echo "Results saved to: ./results/${ENV_NAME}/"
