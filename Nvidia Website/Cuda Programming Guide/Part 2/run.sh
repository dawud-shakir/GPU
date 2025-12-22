#!/usr/bin/env bash
# set -euo pipefail

vectorLength="${1:-1024}"

nvcc -O3 -gencode arch=compute_75,code=sm_75 unified_memory.cu -o unified_memory
nvcc -O3 -gencode arch=compute_75,code=sm_75 explicit_memory.cu -o explicit_memory

./unified_memory "${vectorLength}"
./explicit_memory "${vectorLength}"