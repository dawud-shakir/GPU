#!/usr/bin/env bash
# set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <source.cu>" >&2
    exit 1
fi

file="${1}"

nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" -o "${file%.cu}"