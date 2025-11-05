python multi_host_batch_infer.py \
    --input_dir $1 \
    --output_dir $2 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --base_urls http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1 http://localhost:8004/v1 http://localhost:8005/v1 http://localhost:8006/v1  \
    --max_concurrent 1024
