import argparse
import json
from pathlib import Path
from tqdm import tqdm

def process_file(input_file, prefix_folder, output_file):
    """Process JSON/JSONL file and modify image paths."""
    
    # Determine if input is JSON or JSONL based on extension
    is_jsonl = input_file.endswith('.jsonl')
    
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as f:
        if is_jsonl:
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            # Handle case where JSON file contains a single object
            if isinstance(data, dict):
                data = [data]
    
    # Process each item
    processed_data = []
    for item in tqdm(data, desc="Processing items"):
        if 'image' in item:
            # Get basename of the image path
            basename = Path(item['image']).name
            # Update image path with prefix
            item['image'] = str(Path(prefix_folder) / basename)
        processed_data.append(item)
    
    # Write output as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nProcessed {len(processed_data)} items.")
    print(f"Output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Process JSON/JSONL files and update image paths with a prefix."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Input JSON or JSONL file path"
    )
    parser.add_argument(
        'prefix_folder',
        type=str,
        help="Prefix folder to prepend to image basenames"
    )
    parser.add_argument(
        'output_file',
        type=str,
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    process_file(args.input_file, args.prefix_folder, args.output_file)

if __name__ == "__main__":
    main()
