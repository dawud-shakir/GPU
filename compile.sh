#!/usr/bin/env bash
set -euo pipefail

# ...existing code...
if [ $# -lt 1 ]; then
    echo "Usage: $0 [nvcc-opts...] <source.cu>" >&2
    exit 1
fi

nvcc_args=()
file=""
outfile=""

# Parse args, handle -o <out> or -o<out> and collect other nvcc args
while [ $# -gt 0 ]; do
    case "$1" in
        -o)
            if [ $# -lt 2 ]; then
                echo "Missing argument for -o" >&2
                exit 1
            fi
            outfile="$2"
            shift 2
            ;;
        -o*)
            outfile="${1#-o}"
            shift
            ;;
        *.cu)
            if [ -n "${file}" ]; then
                echo "Multiple .cu source files provided: '${file}' and '$1'" >&2
                exit 1
            fi
            file="$1"
            shift
            ;;
        *)
            nvcc_args+=("$1")
            shift
            ;;
    esac
done

if [ -z "${file}" ]; then
    echo "No .cu source file provided." >&2
    exit 1
fi

# Default output to source base name if -o not given
if [ -z "${outfile}" ]; then
    outfile="${file%.cu}"
fi

echo nvcc -O3 -gencode arch=compute_75,code=sm_75 "${nvcc_args[@]}" "${file}" -o "${outfile}"
nvcc -O3 -gencode arch=compute_75,code=sm_75 "${nvcc_args[@]}" "${file}" -o "${outfile}"


#!/usr/bin/env bash
# set -euo pipefail

# if [ -z "${1:-}" ]; then
#     echo "Usage: $0 <source.cu>" >&2
#     exit 1
# fi

# nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" -o "${file%.cu}"