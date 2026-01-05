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

# If the file path doesn't exist, try kernels/ subfolder (support basename lookup)
if [ ! -f "${file}" ]; then
    if [ -f "kernels/${file}" ]; then
        echo "Using kernels/${file} as source file"
        file="kernels/${file}"
    else
        base="$(basename "${file}")"
        if [ -f "kernels/${base}" ]; then
            echo "Using kernels/${base} as source file"
            file="kernels/${base}"
        else
            echo "Source file '${file}' not found (and not in ./kernels/)." >&2
            exit 1
        fi
    fi
fi

# Default output to source base name if -o not given
if [ -z "${outfile}" ]; then
    outfile="$(basename "${file}" .cu)"
fi

# Helpful nvcc flags:
# nvcc -Xcompiler -Wall -Wextra -Wpedantic
# Enables extra compiler warnings.

# nvcc --resource-usage
# Shows the number of registers and memory used by each kernel.


# Invoke nvcc, but avoid expanding nvcc_args when it's empty/unset
if [ "${#nvcc_args[@]}" -gt 0 ]; then
     echo nvcc -O3 -gencode arch=compute_75,code=sm_75 "${nvcc_args[@]}" "${file}" -o "${outfile}"
     nvcc -O3 -gencode arch=compute_75,code=sm_75 "${nvcc_args[@]}" "${file}" -o "${outfile}"
 else
     echo nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" -o "${outfile}"
     nvcc -O3 -gencode arch=compute_75,code=sm_75 "${file}" -o "${outfile}"
 fi