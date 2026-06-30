#!/usr/bin/env python3
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

def build_prompt(position: str) -> str:
    pos = position.strip().lower()
    if pos in {"middle", "center", "centre"}:
        pos_text = "center"
    elif pos in {"bottom-right", "lower-right", "right-bottom"}:
        pos_text = "bottom-right corner"
    else:
        pos_text = position

    return (
        f"Add a realistic fruit basket in the {pos_text} of the image. "
        "The basket must be clearly visible and contain exactly: one watermelon, "
        "two apples, three peaches, and four bananas. Keep the rest of the image "
        "unchanged, with natural lighting and consistent perspective, high quality."
    )


def dtype_from_str(dtype_str: str):
    key = dtype_str.strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _default_meta_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Meta-attack"))


def main() -> None:
    meta_root = _default_meta_root()
    parser = argparse.ArgumentParser(
        description="Batch edit images using local Qwen-Image-Edit with positional prompts and tqdm."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.getenv("QWEN_IMAGE_EDIT_PATH", "Qwen/Qwen-Image-Edit"),
        help="Local path or HuggingFace ID for Qwen-Image-Edit",
    )

    parser.add_argument(
        "--figstep-dir",
        type=str,
        default=os.path.join(meta_root, "base", "figstep"),
        help="Input directory for figstep images",
    )
    parser.add_argument(
        "--figstep-out",
        type=str,
        default=os.path.join(meta_root, "add_item", "figstep"),
        help="Output directory for edited figstep images",
    )

    parser.add_argument(
        "--mmsafety-dir",
        type=str,
        default=os.path.join(meta_root, "base", "mmsafety"),
        help="Input directory for mmsafety images",
    )
    parser.add_argument(
        "--mmsafety-out",
        type=str,
        default=os.path.join(meta_root, "add_item", "mmsafety"),
        help="Output directory for edited mmsafety images",
    )

    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id to use")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--num-steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--xformers", action="store_true", help="Enable xformers attention if available")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="deformed, blurry, low quality, extra objects, cropped",
        help="Negative prompt text",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow downloading model weights from HuggingFace (default: local_files_only=True)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp,.bmp",
        help="Comma-separated image extensions to include",
    )

    args = parser.parse_args()

    # Set CUDA device visibility before any torch/diffusers CUDA initialization
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    target_dtype = dtype_from_str(args.dtype)

    # Prepare tasks: (input_root, output_root, position)
    tasks: List[Tuple[str, str, str]] = [
        (args.figstep_dir, args.figstep_out, "bottom-right"),
        (args.mmsafety_dir, args.mmsafety_out, "center"),
    ]

    # Collect work items
    ext_tuple = tuple(e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in args.extensions.split(",") if e.strip())
    work_items: List[Dict[str, str]] = []

    for input_root, output_root, position in tasks:
        if not os.path.isdir(input_root):
            # Minimal notice; skip silently if not present
            continue
        images = list_images(input_root, ext_tuple)
        pos_prompt = build_prompt(position)
        for src_path in images:
            rel = os.path.relpath(src_path, input_root)
            dst_path = os.path.join(output_root, rel)
            work_items.append({
                "src": src_path,
                "dst": dst_path,
                "prompt": pos_prompt,
            })

    if len(work_items) == 0:
        # Nothing to do
        return

    # Load pipeline once
    pipeline = QwenImageEditPipeline.from_pretrained(
        args.model_dir,
        local_files_only=not args.online,
    )
    pipeline.to(target_dtype)
    pipeline.to("cuda")
    if args.xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipeline.set_progress_bar_config(disable=True)

    processed = 0
    skipped = 0
    failed = 0

    generator = torch.manual_seed(int(args.seed))

    with tqdm(total=len(work_items), desc="Editing images", dynamic_ncols=True) as pbar:
        for item in work_items:
            src = item["src"]
            dst = item["dst"]
            prompt = item["prompt"]

            if (not args.overwrite) and os.path.exists(dst):
                skipped += 1
                pbar.update(1)
                continue

            ensure_parent_dir(dst)

            try:
                image = Image.open(src).convert("RGB")
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
                    output_image.save(dst)
                processed += 1
            except Exception as e:  # noqa: BLE001 - intentional coarse catch to continue batch
                failed += 1
                # Minimal error line
                tqdm.write(f"Failed: {src} -> {dst} | {e}")
            finally:
                pbar.update(1)

    # Minimal summary
    print(f"done | processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()


