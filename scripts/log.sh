#!/bin/bash
cd $(dirname ${BASH_SOURCE[0]})/..

if [ -d "./log" ]; then
    rm -rf ./log/*
else
    mkdir ./log
fi

funcs="validations stress_conv2d stress_batch_norm stress_leaky_relu stress_maxpool_2d stress_pad"

run_modes="C CUDA CUDAOptimized"
TIMEFORMAT=%R
for run_mode in $run_modes; do
    for func in $funcs; do
        python3 -m lib.test.test $run_mode $func >> ./log/${run_mode}_${func}.log
    done
done