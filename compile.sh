#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 [nvcc-opts...] <source.cu>" >&2
    exit 1
fi

nvcc_args=()
file=""

for arg in "$@"; do
    if [[ "$arg" == *.cu ]]; then
        file="$arg"
    else
        nvcc_args+=("$arg")
    fi
done

if [ -z "${file}" ]; then
    echo "No .cu source file provided." >&2
    exit 1
fi

nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" "${nvcc_args[@]}" -o "${file%.cu}"

#!/usr/bin/env bash
# set -euo pipefail

# if [ -z "${1:-}" ]; then
#     echo "Usage: $0 <source.cu>" >&2
#     exit 1
# fi

# nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" -o "${file%.cu}"