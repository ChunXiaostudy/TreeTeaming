import os
from dataclasses import dataclass

_MOVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Paths:
    project_root: str = _MOVE_ROOT

    meta_root: str = os.path.join(_MOVE_ROOT, "Meta-attack")
    meta_base_root: str = os.path.join(meta_root, "base")
    meta_add_code_root: str = os.path.join(meta_root, "add_code")
    meta_add_item_root: str = os.path.join(meta_root, "add_item")
    meta_prompt_json: str = os.path.join(meta_root, "prompt.json")

    outputs_root: str = os.path.join(_MOVE_ROOT, "outputs")
    generations_root: str = os.path.join(outputs_root, "generations")
    scores_root: str = os.path.join(outputs_root, "scores")


PATHS = Paths()

ADD_ITEM_DATASETS = [
    "add_item_figstep",
    "add_item_mmsafety",
]

META_DATASETS = [
    "add_code_figstep",
    "add_code_mmsafety",
    "add_item_figstep",
    "add_item_mmsafety",
]


def ensure_output_dirs() -> None:
    os.makedirs(PATHS.generations_root, exist_ok=True)
    os.makedirs(PATHS.scores_root, exist_ok=True)
