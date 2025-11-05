#!/bin/bash
for i in {0..6}; do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
      --port $((8000 + i)) \
      --limit-mm-per-prompt.video 0 \
      --gpu-memory-utilization 0.95 \
      --skip-mm-profiling \
      --async-scheduling &
done
wait
