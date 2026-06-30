#!/usr/bin/env python3
"""
Figstep-only object insertion using Qwen-Image-Edit.
De-identified: no personal absolute paths; all IO paths and model path are CLI args.

Example:
  python scripts/add_fruit_basket_figstep.py \
    --model-dir Qwen/Qwen-Image-Edit \
    --input-dir ./data/figstep \
    --output-dir ./outputs/figstep_fruit \
    --dtype bf16 --num-steps 28 --xformers --overwrite
"""
import os
import argparse
from typing import List, Tuple, Dict

from PIL import Image
import torch
from tqdm.auto import tqdm
from diffusers import QwenImageEditPipeline


def list_images(root_dir: str, extensions: Tuple[str, ...]) -> List[str]:
    image_paths: List[str] = []
    for current_root, _dirs, files in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                image_paths.append(os.path.join(current_root, filename))
    image_paths.sort()
    return image_paths


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def dtype_from_str(dtype_str: str):
    key = dtype_str.strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def build_bottom_right_prompt() -> str:
    return (
        "Add a realistic fruit basket in the bottom-right corner of the image. "
        "The basket must be clearly visible and contain exactly: one watermelon, "
        "two apples, three peaches, and four bananas. Keep the rest of the image "
        "unchanged, with natural lighting and consistent perspective, high quality."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a fruit basket to figstep images using Qwen-Image-Edit (de-identified)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="Model ID or local directory for Qwen-Image-Edit",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/figstep",
        help="Input directory containing figstep images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/figstep_fruit",
        help="Output directory for edited images",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--num-steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="deformed, blurry, low quality, extra objects, cropped",
        help="Negative prompt text",
    )
    parser.add_argument("--xformers", action="store_true", help="Enable xformers attention if available")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp,.bmp",
        help="Comma-separated image extensions to include",
    )
    parser.add_argument("--offline", action="store_true", help="Offline mode (HF local files only)")

    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    target_dtype = dtype_from_str(args.dtype)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    images = list_images(
        args.input_dir,
        tuple(e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
              for e in args.extensions.split(",") if e.strip())
    )
    if not images:
        print("no input images; exit")
        return

    pipeline = QwenImageEditPipeline.from_pretrained(
        args.model_dir,
        local_files_only=bool(args.offline),
    )
    pipeline.to(target_dtype)
    pipeline.to(device)
    if args.xformers and device == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipeline.set_progress_bar_config(disable=True)

    prompt = build_bottom_right_prompt()
    generator = torch.manual_seed(int(args.seed))

    processed = 0
    skipped = 0
    failed = 0

    with tqdm(total=len(images), desc="Editing figstep", dynamic_ncols=True) as pbar:
        for src_path in images:
            rel = os.path.relpath(src_path, args.input_dir)
            dst_path = os.path.join(args.output_dir, rel)

            if (not args.overwrite) and os.path.exists(dst_path):
                skipped += 1
                pbar.update(1)
                continue

            ensure_parent_dir(dst_path)
            try:
                image = Image.open(src_path).convert("RGB")
                inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": generator,
                    "true_cfg_scale": float(args.true_cfg_scale),
                    "negative_prompt": str(args.negative_prompt),
                    "num_inference_steps": int(args.num_steps),
                }
                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_image = output.images[0]
                    output_image.save(dst_path)
                processed += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"Failed: {src_path} -> {dst_path} | {e}")
            finally:
                pbar.update(1)

    print(f"done | processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()


