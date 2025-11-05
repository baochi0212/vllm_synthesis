python batch_infer.py \
    --input_dir $1 \
    --output_dir $2 \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --max_concurrent 1024 \
    # --max_tokens 2048
