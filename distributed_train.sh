#!/bin/bash
NUM_PROC=$1
shift

MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT: $MASTER_PORT"

torchrun --nproc_per_node=$NUM_PROC --master_port=$MASTER_PORT train.py "$@"