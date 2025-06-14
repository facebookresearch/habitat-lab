#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_cpus> <command> [args...]"
    exit 1
fi

N=$1
shift

CPUS=$(seq -s, 0 $((N - 1)))

OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N taskset -c $CPUS "$@"
