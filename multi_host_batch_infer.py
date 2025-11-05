import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from PIL import Image
from tqdm import tqdm
from vllm.multimodal.utils import encode_image_base64
import random

# Define prompts
PROMPTS = [
    "Describe this image in detail, including all visible elements, layout, colors, text, and any notable features.",
    # "请详细描述这张图片，包括所有可见元素、布局、颜色、文字以及任何显著特征。",
    # "提供此图像的详细描述。", 
]

def load_and_encode_image(image_path: Path) -> str:
    """Load and encode image (blocking operation)."""
    image = Image.open(image_path)
    return encode_image_base64(image)

async def send_request(
    session: aiohttp.ClientSession,
    image_base64: str,
    prompt: str,
    model: str,
    base_url: str,
    max_tokens: int,
    timeout: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int = None
) -> Dict[str, Any]:
    """Send a single request to the vLLM server."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    
    if seed is not None:
        payload["seed"] = seed
    
    async with session.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def process_image_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    image_path: Path,
    image_base64: str,
    prompt: str,
    model: str,
    base_urls: List[str],
    max_tokens: int,
    timeout: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int = None
) -> Dict[str, Any]:
    """Process a single image with a given prompt using async HTTP request."""
    async with semaphore:  # Control concurrency
        # Select a random base URL for load balancing
        base_url = random.choice(base_urls)
        
        try:
            content = await send_request(
                session=session,
                image_base64=image_base64,
                prompt=prompt,
                model=model,
                base_url=base_url,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed
            )
            return {
                "success": True,
                "content": content,
                "image_path": str(image_path),
                "prompt": prompt,
                "server": base_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path),
                "prompt": prompt,
                "server": base_url
            }

async def test_connection(base_url: str) -> bool:
    """Test connection to vLLM server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return True
                return False
    except Exception as e:
        return False

async def test_all_servers(base_urls: List[str]):
    """Test all server connections."""
    print("Testing connections to vLLM servers...")
    results = await asyncio.gather(*[test_connection(url) for url in base_urls])
    
    active_servers = []
    for url, is_active in zip(base_urls, results):
        if is_active:
            print(f"✓ {url} - Active")
            active_servers.append(url)
        else:
            print(f"✗ {url} - Unreachable")
    
    print()
    return active_servers

async def process_batch_async(
    image_paths: List[Path],
    output_dir: Path,
    model: str,
    base_urls: List[str],
    max_tokens: int,
    timeout: int,
    max_concurrent: int,
    prompts: List[str],
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int = None
):
    """Process multiple images concurrently with progress tracking."""
    
    # Test all server connections
    active_servers = await test_all_servers(base_urls)
    
    if not active_servers:
        print(f"❌ No vLLM servers are reachable")
        print("Please check if the servers are running and accessible")
        return
    
    print(f"Using {len(active_servers)} active servers\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-load and encode all images (blocking operations in thread pool)
    print("Loading and encoding images...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        image_base64_futures = [
            loop.run_in_executor(executor, load_and_encode_image, path)
            for path in image_paths
        ]
        image_base64_list = []
        for future in tqdm(asyncio.as_completed(image_base64_futures), total=len(image_base64_futures)):
            image_base64_list.append(await future)
    print(f"✓ Encoded {len(image_base64_list)} images\n")
    
    # Create semaphore for concurrency control
    # With multiple servers, we can handle more concurrent requests
    effective_concurrency = max_concurrent * len(active_servers)
    semaphore = asyncio.Semaphore(effective_concurrency)
    print(f"Effective concurrency: {effective_concurrency} ({max_concurrent} per server × {len(active_servers)} servers)\n")
    
    # Prepare all tasks
    tasks = []
    image_info = []
    
    async with aiohttp.ClientSession() as session:
        for idx, (image_path, image_base64) in enumerate(zip(image_paths, image_base64_list)):
            for prompt_idx, prompt in enumerate(prompts):
                select_prompt_idx = random.choice(range(len(prompts)))
                if prompt_idx != select_prompt_idx:
                    continue
                task = process_image_async(
                    session=session,
                    semaphore=semaphore,
                    image_path=image_path,
                    image_base64=image_base64,
                    prompt=prompt,
                    model=model,
                    base_urls=active_servers,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed
                )
                tasks.append(task)
                image_info.append({
                    "image_idx": idx,
                    "image_path": image_path,
                    "prompt_idx": prompt_idx
                })
        
        # Execute all tasks with progress bar
        print("Processing requests...")
        results = []
        with tqdm(total=len(tasks), desc="API requests") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
    
    # Match results back to their original order
    result_map = {}
    for info, result in zip(image_info, results):
        key = (info["image_idx"], info["prompt_idx"])
        result_map[key] = (info, result)
    
    # Organize results by image in original order
    image_results = {}
    for info, result in zip(image_info, results):
        img_idx = info["image_idx"]
        if img_idx not in image_results:
            image_results[img_idx] = {
                "image_path": info["image_path"],
                "results": []
            }
        key = (info["image_idx"], info["prompt_idx"])
        if key in result_map:
            _, result = result_map[key]
            image_results[img_idx]["results"].append({
                "prompt_idx": info["prompt_idx"],
                "prompt": prompts[info["prompt_idx"]],
                "result": result
            })
    
    # Convert to LLaVA format and save
    llava_data = []
    failed_count = 0
    server_stats = {url: 0 for url in active_servers}
    
    for img_idx, data in sorted(image_results.items()):
        image_path = data["image_path"]
        
        # Create LLaVA format entries for each prompt
        for prompt_result in data["results"]:
            if prompt_result["result"]["success"]:
                llava_entry = {
                    "id": f"{image_path.stem}_{img_idx:06d}_prompt_{prompt_result['prompt_idx']}",
                    "image": str(image_path),
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{prompt_result['prompt']}"
                        },
                        {
                            "from": "gpt",
                            "value": prompt_result["result"]["content"]
                        }
                    ]
                }
                llava_data.append(llava_entry)
                server_stats[prompt_result["result"]["server"]] += 1
            else:
                failed_count += 1
                print(f"Failed: {image_path} on {prompt_result['result'].get('server', 'unknown')} - {prompt_result['result'].get('error', 'Unknown error')}")
    
    # Save LLaVA format JSON
    output_json_path = output_dir / "llava_format.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(llava_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Processed {len(image_paths)} images with {len(prompts)} prompts each")
    print(f"✓ Saved {len(llava_data)} successful entries to {output_json_path}")
    if failed_count > 0:
        print(f"⚠ {failed_count} requests failed")
    
    print("\nServer load distribution:")
    for server, count in server_stats.items():
        print(f"  {server}: {count} requests")

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference script for vLLM with concurrent requests and multi-server support"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-32B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--base_urls",
        type=str,
        nargs="+",
        default=["http://localhost:8000/v1"],
        help="vLLM server base URLs (space-separated for multiple servers)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent requests PER SERVER"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy/deterministic decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p (1.0 to disable)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling (-1 to disable)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".webp"],
        help="Image file extensions to process"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Get all image files
    image_paths = []
    for ext in args.image_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print(f"Error: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Using {len(PROMPTS)} prompts per image")
    print(f"Total requests: {len(PROMPTS)}")
    print(f"Configured servers: {len(args.base_urls)}")
    print(f"Max concurrent per server: {args.max_concurrent}")
    print(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()
    
    # Run batch processing
    start_time = time.time()
    asyncio.run(process_batch_async(
        image_paths=image_paths,
        output_dir=output_dir,
        model=args.model,
        base_urls=args.base_urls,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        prompts=PROMPTS,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed
    ))
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time / len(image_paths):.2f}s")
    print(f"Throughput: {len(image_paths) * len(PROMPTS) / elapsed_time:.2f} requests/sec")

if __name__ == "__main__":
    main()
