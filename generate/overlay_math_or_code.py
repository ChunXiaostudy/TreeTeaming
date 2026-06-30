#!/usr/bin/env python3
import os
import io
import re
import argparse
import random
from typing import List, Tuple, Dict, Optional

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# Matplotlib (CPU-only, offscreen)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


def parse_math_md(md_path: str) -> Tuple[List[str], str]:
    with open(md_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    formulas: List[str] = []
    code_lines: List[str] = []

    # Extract 10 formulas (lines 1..10 style) and code block after a line starting with 'code:'
    code_mode = False
    for line in lines:
        if code_mode:
            code_lines.append(line)
            continue
        if re.match(r"^\s*code\s*:\s*$", line, flags=re.IGNORECASE):
            code_mode = True
            continue

        # Attempt to strip a leading \text{...} and keep the math content
        if line.strip().startswith("\\text{"):
            stripped = re.sub(r"^\\text\{[^}]*\}\s*", "", line.strip())
            if stripped:
                formulas.append(stripped)
        elif line.strip():
            # Fallback: if the line looks like a formula, keep it
            formulas.append(line.strip())

    # Keep only first 10 formulas if more were accidentally parsed
    formulas = formulas[:10]
    # Join code block lines after 'code:' marker
    code_text = "\n".join(code_lines).strip()
    return formulas, code_text


def render_text_rgba(
    text: str,
    fontsize: int = 36,
    dpi: int = 220,
    text_color: str = "#FFFFFF",
    stroke_width: float = 4.0,
    stroke_color: str = "#000000",
    font_family: str = "DejaVu Sans",
) -> Image.Image:
    # Render multiline text (supporting mathtext via $...$) to RGBA image with transparent background
    # Strategy: draw on a large enough canvas, then crop by alpha
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    txt = ax.text(
        0.02,
        0.98,
        text,
        color=text_color,
        fontsize=fontsize,
        family=font_family,
        va="top",
        ha="left",
    )
    txt.set_path_effects([
        path_effects.withStroke(linewidth=stroke_width, foreground=stroke_color)
    ])

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA")

    # Auto-crop by alpha
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            left, right = int(xs.min()), int(xs.max())
            top, bottom = int(ys.min()), int(ys.max())
            img = img.crop((left, top, right + 1, bottom + 1))

    plt.close(fig)
    return img


def render_formulas_block(formulas: List[str], fontsize: int = 36, dpi: int = 220, stroke_width: float = 4.0) -> Image.Image:
    # Wrap each formula with $...$ for mathtext
    lines = [f"${f.strip()}$" for f in formulas]
    text = "\n\n".join(lines)
    return render_text_rgba(text=text, fontsize=fontsize, dpi=dpi, stroke_width=stroke_width)


def render_code_block(code_text: str, fontsize: int = 28, dpi: int = 220, stroke_width: float = 4.0) -> Image.Image:
    # Try pygments first for nice syntax highlighting; fallback to simple text rendering
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import ImageFormatter

        formatter = ImageFormatter(
            font_name="DejaVu Sans Mono",
            font_size=fontsize,
            line_numbers=False,
            image_format="PNG",
            style="default",
        )
        data = highlight(code_text, PythonLexer(), formatter)
        image = Image.open(io.BytesIO(data)).convert("RGBA")
        return image
    except Exception:
        # Fallback: render as plain monospaced text
        return render_text_rgba(text=code_text, fontsize=fontsize, dpi=dpi, font_family="DejaVu Sans Mono", stroke_width=stroke_width)


def paste_overlay(
    base: Image.Image,
    overlay: Image.Image,
    position: str,
    max_width_frac: float = 0.5,
    margin: int = 16,
    bg_alpha: float = 0.28,
    bg_pad: int = 8,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    xy: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    base = base.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Resize overlay to fit within fraction of base width
    max_w = int(base.width * max_width_frac)
    if overlay.width > max_w and max_w > 0:
        scale = max_w / float(overlay.width)
        new_w = max(1, int(round(overlay.width * scale)))
        new_h = max(1, int(round(overlay.height * scale)))
        overlay = overlay.resize((new_w, new_h), Image.LANCZOS)

    if xy is not None:
        x, y = int(xy[0]), int(xy[1])
    elif position == "bottom-right":
        x = max(margin, base.width - overlay.width - margin)
        y = max(margin, base.height - overlay.height - margin)
    elif position == "top":
        x = max(0, (base.width - overlay.width) // 2)
        y = max(margin, margin)
    elif position == "center":
        x = max(0, (base.width - overlay.width) // 2)
        y = max(0, (base.height - overlay.height) // 2)
    else:
        x, y = margin, margin

    out = base.copy()
    if bg_alpha > 0.0:
        bg_w = overlay.width + 2 * int(bg_pad)
        bg_h = overlay.height + 2 * int(bg_pad)
        rect = Image.new("RGBA", (bg_w, bg_h), (bg_color[0], bg_color[1], bg_color[2], int(255 * float(bg_alpha))))
        rx = max(0, x - int(bg_pad))
        ry = max(0, y - int(bg_pad))
        out.alpha_composite(rect, dest=(rx, ry))
    out.alpha_composite(overlay, dest=(x, y))
    return out.convert("RGB")


def compute_saliency_map_gray(gray_arr: np.ndarray) -> np.ndarray:
    # gray_arr: float32 in [0,1]
    gy, gx = np.gradient(gray_arr)
    mag = np.hypot(gx, gy)
    mmax = float(mag.max()) if mag.size > 0 else 0.0
    if mmax > 0:
        mag = mag / mmax
    return mag


def find_least_salient_position(
    base: Image.Image,
    overlay: Image.Image,
    search_rows: int = 5,
    search_cols: int = 5,
    center_region_frac: float = 0.7,
) -> Tuple[int, int]:
    # Downscale for speed
    bw, bh = base.size
    scale = 1.0
    max_side = max(bw, bh)
    if max_side > 512:
        scale = 512.0 / float(max_side)
    sw = max(1, int(round(bw * scale)))
    sh = max(1, int(round(bh * scale)))
    small = base.convert("L").resize((sw, sh), Image.BILINEAR)
    gray = np.asarray(small, dtype=np.float32) / 255.0
    sal = compute_saliency_map_gray(gray)

    ow, oh = overlay.size
    osw = max(1, int(round(ow * scale)))
    osh = max(1, int(round(oh * scale)))

    # Center-constrained search region
    reg_w = min(sw, int(round(sw * center_region_frac)))
    reg_h = min(sh, int(round(sh * center_region_frac)))
    cx, cy = sw // 2, sh // 2
    rx0 = max(0, cx - reg_w // 2)
    ry0 = max(0, cy - reg_h // 2)
    rx1 = min(sw, rx0 + reg_w)
    ry1 = min(sh, ry0 + reg_h)

    # Ensure overlay fits inside region; if not, fallback to strict center
    if osw >= (rx1 - rx0) or osh >= (ry1 - ry0):
        x = (bw - ow) // 2
        y = (bh - oh) // 2
        return int(max(0, x)), int(max(0, y))

    # Grid sampling
    xs = [int(round(rx0 + i * ((rx1 - rx0 - osw) / max(1, search_cols - 1)))) for i in range(search_cols)]
    ys = [int(round(ry0 + j * ((ry1 - ry0 - osh) / max(1, search_rows - 1)))) for j in range(search_rows)]

    best_score = None
    best_xy_small = (max(0, (sw - osw) // 2), max(0, (sh - osh) // 2))
    for yy in ys:
        for xx in xs:
            window = sal[yy:yy + osh, xx:xx + osw]
            if window.size == 0:
                continue
            score = float(window.mean())
            if (best_score is None) or (score < best_score):
                best_score = score
                best_xy_small = (xx, yy)

    # Map back to original scale
    x = int(round(best_xy_small[0] / scale))
    y = int(round(best_xy_small[1] / scale))
    # Clamp to boundaries
    x = max(0, min(x, bw - ow))
    y = max(0, min(y, bh - oh))
    return x, y


def _default_meta_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Meta-attack"))


def main() -> None:
    meta_root = _default_meta_root()
    parser = argparse.ArgumentParser(description="Overlay random math formulas or code snippet onto images (CPU-only).")
    parser.add_argument(
        "--math-md",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "math.md"),
        help="Path to math.md containing 10 formulas and a code block",
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
        default=os.path.join(meta_root, "add_code", "figstep"),
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
        default=os.path.join(meta_root, "add_code", "mmsafety"),
        help="Output directory for edited mmsafety images",
    )
    parser.add_argument(
        "--prob-code",
        type=float,
        default=0.5,
        help="Probability of placing the code snippet instead of 3 random formulas",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.webp,.bmp",
        help="Comma-separated image extensions to include",
    )
    parser.add_argument("--fontsize", type=int, default=36, help="Font size for rendering")
    parser.add_argument("--dpi", type=int, default=220, help="DPI for text rendering")
    parser.add_argument("--stroke-width", type=float, default=4.0, help="Stroke width for text outline")
    parser.add_argument("--width-frac-br", type=float, default=0.66, help="Max overlay width fraction (bottom-right)")
    parser.add_argument("--width-frac-center", type=float, default=0.72, help="Max overlay width fraction (center)")
    parser.add_argument("--margin", type=int, default=16, help="Margin in pixels for bottom-right placement")
    parser.add_argument("--bg-alpha", type=float, default=0.28, help="Background rectangle alpha (0..1), 0 disables")
    parser.add_argument("--bg-pad", type=int, default=8, help="Padding around overlay inside background rectangle")
    parser.add_argument("--avoid-occlusion", action="store_true", default=True, help="Avoid occluding salient regions for center placement")
    parser.add_argument("--search-rows", type=int, default=5, help="Rows in center search grid")
    parser.add_argument("--search-cols", type=int, default=5, help="Cols in center search grid")

    args = parser.parse_args()

    random.seed(int(args.seed))

    # Prepare tasks: (input_root, output_root, position)
    tasks: List[Tuple[str, str, str]] = [
        (args.figstep_dir, args.figstep_out, "bottom-right"),
        (args.mmsafety_dir, args.mmsafety_out, "top"),
    ]

    formulas, code_text = parse_math_md(args.math_md)

    ext_tuple = tuple(e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in args.extensions.split(",") if e.strip())
    work_items: List[Dict[str, str]] = []
    for input_root, output_root, position in tasks:
        if not os.path.isdir(input_root):
            continue
        images = list_images(input_root, ext_tuple)
        for src_path in images:
            rel = os.path.relpath(src_path, input_root)
            dst_path = os.path.join(output_root, rel)
            work_items.append({
                "src": src_path,
                "dst": dst_path,
                "position": position,
            })

    if len(work_items) == 0:
        return

    processed = 0
    skipped = 0
    failed = 0

    with tqdm(total=len(work_items), desc="Overlaying", dynamic_ncols=True) as pbar:
        for item in work_items:
            src = item["src"]
            dst = item["dst"]
            position = item["position"]

            if (not args.overwrite) and os.path.exists(dst):
                skipped += 1
                pbar.update(1)
                continue

            ensure_parent_dir(dst)

            try:
                base = Image.open(src).convert("RGBA")

                # Random choice: code vs 3 formulas
                use_code = (random.random() < float(args.prob_code)) and bool(code_text)
                if use_code:
                    overlay = render_code_block(code_text, fontsize=args.fontsize, dpi=args.dpi, stroke_width=float(args.stroke_width))
                else:
                    chosen = random.sample(formulas, k=min(3, len(formulas))) if formulas else []
                    overlay = render_formulas_block(chosen, fontsize=args.fontsize, dpi=args.dpi, stroke_width=float(args.stroke_width))

                if position == "center" and args.avoid_occlusion:
                    # Pre-resize overlay to match center width fraction before choosing position
                    max_w = int(base.width * float(args.width_frac_center))
                    overlay_res = overlay
                    if overlay.width > max_w and max_w > 0:
                        scale = max_w / float(overlay.width)
                        new_w = max(1, int(round(overlay.width * scale)))
                        new_h = max(1, int(round(overlay.height * scale)))
                        overlay_res = overlay.resize((new_w, new_h), Image.LANCZOS)

                    x, y = find_least_salient_position(
                        base=base,
                        overlay=overlay_res,
                        search_rows=int(args.search_rows),
                        search_cols=int(args.search_cols),
                        center_region_frac=0.7,
                    )
                    out = paste_overlay(
                        base=base,
                        overlay=overlay_res,
                        position=position,
                        max_width_frac=1.0,  # already resized
                        margin=int(args.margin),
                        bg_alpha=float(args.bg_alpha),
                        bg_pad=int(args.bg_pad),
                        xy=(x, y),
                    )
                else:
                    width_frac = args.width_frac_br if position == "bottom-right" else args.width_frac_center
                    local_bg_alpha = float(args.bg_alpha)
                    if position == "top":
                        local_bg_alpha = min(local_bg_alpha, 0.18)
                    out = paste_overlay(
                        base=base,
                        overlay=overlay,
                        position=position,
                        max_width_frac=float(width_frac),
                        margin=int(args.margin),
                        bg_alpha=local_bg_alpha,
                        bg_pad=int(args.bg_pad),
                    )
                out.save(dst)
                processed += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"Failed: {src} -> {dst} | {e}")
            finally:
                pbar.update(1)

    print(f"done | processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()


