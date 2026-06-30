import os
import json
from typing import List, Tuple

from tqdm import tqdm

from .config import PATHS, ensure_output_dirs
from .closed_infer import _read_image_as_data_url, _call_openai_chat


def _out_jsonl_path(model_type: str, dataset: str) -> str:
    sub = os.path.join(PATHS.generations_root, dataset)
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{model_type}.jsonl")


def _load_existing_images(jsonl_path: str) -> set:
    exists = set()
    if not os.path.isfile(jsonl_path):
        return exists
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                img = rec.get('image')
                if isinstance(img, str):
                    exists.add(img)
    except Exception:
        return exists
    return exists


def _iter_meta_images() -> List[Tuple[str, str, str]]:
    """返回 (dataset_name, image_path, prompt) 列表。
    dataset_name in {add_code_figstep, add_code_mmsafety, add_item_figstep, add_item_mmsafety}
    figstep 目录支持 figstep 或 fig-step。
    prompt 取自 prompt.json 的四个字段：add_code_figstep、add_code_mmsafety、add_item_figstep、add_item_mmsafety。
    """
    path_prompt = PATHS.meta_prompt_json
    if not os.path.isfile(path_prompt):
        return []
    with open(path_prompt, 'r', encoding='utf-8') as f:
        prompt_obj = json.load(f)
    # 读取四类 prompt，兼容旧键名回退
    p_add_item_fig = str(prompt_obj.get('add_item_figstep', prompt_obj.get('add_item', ''))).strip()
    p_add_item_mm  = str(prompt_obj.get('add_item_mmsafety', prompt_obj.get('add_item', ''))).strip()
    p_add_code_fig = str(prompt_obj.get('add_code_figstep', prompt_obj.get('add_code', ''))).strip()
    p_add_code_mm  = str(prompt_obj.get('add_code_mmsafety', prompt_obj.get('add_code', ''))).strip()
    prompts_map = {
        'add_item_figstep': p_add_item_fig,
        'add_item_mmsafety': p_add_item_mm,
        'add_code_figstep': p_add_code_fig,
        'add_code_mmsafety': p_add_code_mm,
    }
    # 至少需要存在任意一个 prompt
    if not any(prompts_map.values()):
        return []

    def _collect(dir_root: str, sub: str) -> List[str]:
        p1 = os.path.join(dir_root, 'figstep')
        p2 = os.path.join(dir_root, 'fig-step')
        fig_dir = p1 if os.path.isdir(p1) else p2
        mm_dir = os.path.join(dir_root, 'mmsafety')
        paths = []
        if os.path.isdir(fig_dir):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    paths.append((f"{sub}_figstep", os.path.join(fig_dir, fn)))
        if os.path.isdir(mm_dir):
            for fn in sorted(os.listdir(mm_dir)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    paths.append((f"{sub}_mmsafety", os.path.join(mm_dir, fn)))
        return paths

    items_code = _collect(PATHS.meta_add_code_root, 'add_code')
    items_item = _collect(PATHS.meta_add_item_root, 'add_item')

    results: List[Tuple[str, str, str]] = []
    for dataset_name, img_path in items_code:
        prompt = prompts_map.get(dataset_name, '')
        if not prompt:
            # 回退：若子类缺失，尝试使用对应 add_code 的另一子类或旧键
            prompt = p_add_code_fig if dataset_name.endswith('figstep') else p_add_code_mm
        results.append((dataset_name, img_path, prompt))
    for dataset_name, img_path in items_item:
        prompt = prompts_map.get(dataset_name, '')
        if not prompt:
            prompt = p_add_item_fig if dataset_name.endswith('figstep') else p_add_item_mm
        results.append((dataset_name, img_path, prompt))
    return results


def generate_meta_closed(model_list: List[str], only_datasets: List[str] | None = None) -> None:
    ensure_output_dirs()
    items = _iter_meta_images()
    if not items:
        return
    if only_datasets:
        only_set = set(only_datasets)
        items = [(ds, p, q) for (ds, p, q) in items if ds in only_set]
        if not items:
            return
    # 分 dataset 写入，断点续跑
    datasets = sorted(list({ds for ds, _p, _q in items}))
    for model_name in model_list:
        for dataset in datasets:
            out_path = _out_jsonl_path(model_name, dataset)
            existing = _load_existing_images(out_path)
            subset = [(ds, p, q) for (ds, p, q) in items if ds == dataset and p not in existing]
            if not subset:
                continue
            with open(out_path, 'a', encoding='utf-8', buffering=1) as f:
                for _ds, image_path, prompt in tqdm(subset, desc=f"{model_name} @ {dataset}"):
                    try:
                        data_url = _read_image_as_data_url(image_path)
                        messages = [
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": prompt},
                            ]}
                        ]
                        response = _call_openai_chat(model_name, messages)
                    except Exception as e:
                        response = f"<gen_error:{type(e).__name__}:{e}>"
                    record = {
                        'model': model_name,
                        'dataset': dataset,
                        'image': image_path,
                        'response': response,
                        'meta': {},
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='gpt-4o-2024-05-13,claude-3-5-sonnet-20240620')
    parser.add_argument('--only', type=str, default='', help='comma-separated subset: add_item_figstep,add_item_mmsafety,...')
    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    only_list = [s.strip() for s in args.only.split(',') if s.strip()] if args.only else None
    generate_meta_closed(model_list, only_list)


