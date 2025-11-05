#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Iterable, Tuple

from tqdm import tqdm

# Optional speed-ups:
# 1) Enable hf_transfer uploader (pip install hf_transfer) for faster uploads
# 2) Multi-threaded uploads for zip mode (upload_folder max_workers)
def maybe_enable_hf_transfer(enable: bool):
    if not enable:
        return
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Enabled hf_transfer uploader.", flush=True)
    except Exception:
        print("hf_transfer not installed. Install with: pip install hf-transfer", flush=True)

def human_readable_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def iter_image_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def compute_total_size(files: List[Path]) -> int:
    total = 0
    for p in tqdm(files, desc="Scanning sizes", unit="file"):
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total

def partition_by_target_size(files: List[Path], max_bytes: int) -> List[List[Path]]:
    shards: List[List[Path]] = []
    current, acc = [], 0
    for p in files:
        size = 0
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            continue
        # If a single file exceeds max_bytes, put it alone in a shard
        if size >= max_bytes:
            if current:
                shards.append(current)
                current, acc = [], 0
            shards.append([p])
            continue
        if acc + size > max_bytes and current:
            shards.append(current)
            current, acc = [], 0
        current.append(p)
        acc += size
    if current:
        shards.append(current)
    return shards

def zip_shards(shards: List[List[Path]], dest_dir: Path, compression: str = "stored") -> List[Path]:
    import zipfile

    comp = zipfile.ZIP_STORED if compression.lower() == "stored" else zipfile.ZIP_DEFLATED
    out_paths: List[Path] = []

    for idx, shard in enumerate(shards):
        out_path = dest_dir / f"images-shard-{idx:05d}.zip"
        with zipfile.ZipFile(out_path, mode="w", compression=comp, compresslevel=0 if comp == zipfile.ZIP_STORED else None) as zf:
            for p in tqdm(shard, desc=f"Zipping shard {idx}", unit="file"):
                # Store relative paths to preserve any class subfolders
                arcname = p.relative_to(shard[0].parents[0]) if len(shard) > 0 else p.name
                # If above relative_to fails in some layouts, fallback to filename only
                try:
                    zf.write(p, arcname=str(arcname))
                except Exception:
                    zf.write(p, arcname=p.name)
        out_paths.append(out_path)
    return out_paths

def ensure_hf_login(token: str | None):
    if token:
        try:
            from huggingface_hub import HfFolder
            HfFolder.save_token(token)
        except Exception as e:
            print(f"Warning: could not save HF token locally: {e}", flush=True)

def create_or_get_repo(repo_id: str, private: bool, token: str | None):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    except Exception as e:
        print(f"Warning: create_repo error (may already exist): {e}", flush=True)
    return api

def upload_zip_folder(api, repo_id: str, local_dir: Path, token: str | None, max_workers: int):
    from huggingface_hub import upload_folder
    upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        max_workers=max_workers,
        allow_patterns=["*.zip", "README.md", "dataset_card.md"],
    )

def parquet_mode(args: argparse.Namespace):
    # Fast, Hub-native parquet shards with datasets.push_to_hub
    maybe_enable_hf_transfer(args.enable_hf_transfer)
    ensure_hf_login(args.token)
    _ = create_or_get_repo(args.repo_id, args.private, args.token)

    # Load as imagefolder; this keeps file references and lets push_to_hub upload the files + parquet
    from datasets import load_dataset
    print("Indexing images with datasets (imagefolder)...", flush=True)
    ds_dict = load_dataset(
        "imagefolder",
        data_dir=args.input_dir,
        split="train",
        drop_labels=False,   # Keep class labels if you're using class-subfolders
    )

    # Push with sharding; leave headroom below 10GB
    print(f"Pushing to {args.repo_id} with parquet shards <= {args.max_shard_size} ...", flush=True)
    ds_dict.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        max_shard_size=args.max_shard_size,  # e.g., "9GB"
    )
    print("Done.", flush=True)

def zip_mode(args: argparse.Namespace):
    maybe_enable_hf_transfer(args.enable_hf_transfer)
    ensure_hf_login(args.token)
    api = create_or_get_repo(args.repo_id, args.private, args.token)

    root = Path(args.input_dir).expanduser().resolve()
    files = list(iter_image_files(root, recursive=True))
    if not files:
        print("No images found. Supported extensions: " + ", ".join(sorted(IMAGE_EXTS)), flush=True)
        sys.exit(1)

    total_size = compute_total_size(files)
    print(f"Found {len(files)} images, total size ≈ {human_readable_size(total_size)}", flush=True)

    max_bytes = args.zip_max_bytes
    shards = partition_by_target_size(files, max_bytes)
    print(f"Creating {len(shards)} zip shards (target per-shard <= {human_readable_size(max_bytes)}) ...", flush=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="hf_zip_shards_"))
    try:
        zips = zip_shards(shards, tmpdir, compression=args.zip_compression)
        # Optional dataset card for basic guidance
        (tmpdir / "README.md").write_text(
            "# Zipped image shards\n\n"
            "This dataset consists of multiple .zip shards uploaded for efficient hosting.\n"
            "You can stream with `datasets` using `data_files` patterns:\n\n"
            "from datasets import load_dataset\n"
            "ds = load_dataset('imagefolder', data_files='*.zip', split='train', streaming=True)\n",
            encoding="utf-8",
        )
        print(f"Uploading {len(zips)} shards to {args.repo_id} ...", flush=True)
        upload_zip_folder(api, args.repo_id, tmpdir, args.token, max_workers=args.max_workers)
        print("Done.", flush=True)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partition an image folder and upload to Hugging Face Datasets.")
    p.add_argument("--input_dir", required=True, help="Path to folder containing images (optionally with class subfolders).")
    p.add_argument("--repo_id", required=True, help="Target Hugging Face dataset repo, e.g. username/dataset_name")
    p.add_argument("--private", action="store_true", help="Create/upload to a private dataset repo.")
    p.add_argument("--token", default=None, help="HF token (else takes from env HF_TOKEN).")

    # Mode
    p.add_argument("--format", choices=["parquet", "zip"], default="parquet",
                   help="Parquet uses datasets.push_to_hub with sharded parquet. Zip creates .zip shards and uploads them.")

    # Parquet mode options
    p.add_argument("--max_shard_size", default="9GB",
                   help="Parquet shard size for push_to_hub (string like '9GB', keep below 10GB).")

    # Zip mode options
    p.add_argument("--zip_max_bytes", type=int, default=int(9.2 * (1024**3)),
                   help="Max bytes per zip shard (default ~9.2GB for headroom below 10GB).")
    p.add_argument("--zip_compression", choices=["stored", "deflate"], default="stored",
                   help="Zip compression: 'stored' is fastest (no compression), 'deflate' compresses but slower.")
    p.add_argument("--max_workers", type=int, default=8, help="Parallel upload workers for zip mode.")
    p.add_argument("--keep_temp", action="store_true", help="Keep temporary zip directory after upload.")

    # Speed-up
    p.add_argument("--enable_hf_transfer", action="store_true",
                   help="Enable hf_transfer uploader for faster uploads (pip install hf-transfer).")

    return p.parse_args()

def main():
    args = parse_args()
    # Validate repo_id
    if "/" not in args.repo_id:
        print("repo_id must be like 'username/dataset_name'", flush=True)
        sys.exit(1)

    # Route to mode
    if args.format == "parquet":
        parquet_mode(args)
    else:
        zip_mode(args)

if __name__ == "__main__":
    main()
