#!/usr/bin/env python3
"""从 FigStep / MM-SafetyBench 原始数据复制 base 图像到 Meta-attack/base/。"""
import argparse
import os
import shutil
from typing import List, Tuple


def list_images(root_dir: str, extensions: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    for current_root, _dirs, files in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                paths.append(os.path.join(current_root, filename))
    paths.sort()
    return paths


def copy_tree(src_root: str, dst_root: str, extensions: Tuple[str, ...], overwrite: bool) -> int:
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source directory not found: {src_root}")
    os.makedirs(dst_root, exist_ok=True)
    copied = 0
    for src in list_images(src_root, extensions):
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_root = os.path.abspath(os.path.join(script_dir, "..", "Meta-attack"))

    parser = argparse.ArgumentParser(description="Prepare Meta-attack base images from upstream datasets.")
    parser.add_argument(
        "--figstep-src",
        type=str,
        required=True,
        help="FigStep SafeBench images root (e.g. FigStep-main/data/images/SafeBench)",
    )
    parser.add_argument(
        "--mmsafety-src",
        type=str,
        required=True,
        help="MM-SafetyBench images root (e.g. MM-SafetyBench(imgs))",
    )
    parser.add_argument(
        "--figstep-dst",
        type=str,
        default=os.path.join(meta_root, "base", "figstep"),
    )
    parser.add_argument(
        "--mmsafety-dst",
        type=str,
        default=os.path.join(meta_root, "base", "mmsafety"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp,.bmp",
    )
    args = parser.parse_args()

    ext_tuple = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in args.extensions.split(",")
        if e.strip()
    )

    n_fig = copy_tree(args.figstep_src, args.figstep_dst, ext_tuple, args.overwrite)
    n_mm = copy_tree(args.mmsafety_src, args.mmsafety_dst, ext_tuple, args.overwrite)
    print(f"done | figstep={n_fig} -> {args.figstep_dst} | mmsafety={n_mm} -> {args.mmsafety_dst}")


if __name__ == "__main__":
    main()
