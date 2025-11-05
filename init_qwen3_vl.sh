# vllm serve Qwen/Qwen3-VL-32B-Instruct \
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  # --max-model-len 8192 


