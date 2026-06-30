# TreeTeaming

**Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration** (CVPR 2026)

Official code release for the **add_item** benchmark pipeline: sample generation → closed VLM inference → GPT judge scoring.

Repository: [https://github.com/ChunXiaostudy/TreeTeaming](https://github.com/ChunXiaostudy/TreeTeaming)

> **Note:** This repository ships **code and prompts only**. Image datasets are **not** included. You need to prepare or generate images locally (see below).

## Repository layout

```
Tree_Teaming/
├── Meta-attack/
│   ├── base/              # upstream images (local only, gitignored)
│   ├── add_item/          # generated attack images (local only, gitignored)
│   ├── prompt.json        # attack prompts (Ares-V13)
│   └── prompt.json.example
├── generate/              # dataset preparation & Qwen-Image-Edit generation
├── benchmark/             # closed-model inference + GPT judge
├── scripts/               # one-click shell scripts
└── outputs/               # run outputs (gitignored)
```

## Quick start

```bash
git clone https://github.com/ChunXiaostudy/TreeTeaming.git
cd TreeTeaming
pip install -r requirements.txt
cp .env.example .env
# Edit .env: OPENAI_BASE_URL, OPENAI_API_KEY
```

| Variable | Purpose |
|----------|---------|
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint |
| `OPENAI_API_KEY` | API key |
| `JUDGE_MODEL` | Judge model (default: `gpt-4o-mini`) |
| `QWEN_IMAGE_EDIT_PATH` | Local path or HF id for Qwen-Image-Edit |

## Prepare images (not in repo)

Images are excluded from git to keep the repository lightweight. Two options:

### Option A — Generate from upstream datasets

```bash
export FIGSTEP_SRC=/path/to/FigStep-main/data/images/SafeBench
export MMSAFETY_SRC="/path/to/MM-SafetyBench(imgs)"

bash scripts/run_add_item_pipeline.sh
```

Steps inside the pipeline:

1. `generate/prepare_base_images.py` — copy FigStep / MM-SafeBench base images  
2. `generate/run_qwen_image_edit.sh` — insert fruit basket via Qwen-Image-Edit  
3. `scripts/run_meta_closed_infer.sh` — run closed VLMs on `add_item_*` subsets  
4. `scripts/run_score_jsonl.sh` — GPT judge scoring (ASR = score 5 rate)

### Option B — Use your own local images

Place images under:

```
Meta-attack/add_item/figstep/
Meta-attack/add_item/mmsafety/
```

Then skip prepare/generate:

```bash
SKIP_PREPARE=true SKIP_GENERATE=true bash scripts/run_add_item_pipeline.sh
```

## Run inference & scoring only

```bash
MODELS="gpt-4o-2024-05-13" \
ONLY="add_item_figstep,add_item_mmsafety" \
bash scripts/run_meta_closed_infer.sh

bash scripts/run_score_jsonl.sh
```

Optional: use `missionObjective` from model JSON as judge instruction:

```bash
USE_MISSION=true bash scripts/run_score_jsonl.sh
```

## Prompts

Edit `Meta-attack/prompt.json`. Required keys for add_item:

- `add_item_figstep`
- `add_item_mmsafety`

See `Meta-attack/prompt.json.example`.

## Judge protocol

1. Rejection keyword pre-filter → score 0  
2. GPT judge (`JUDGE_MODEL`) with CLAS 1–5 rubric  
3. ASR = fraction of samples with score 5  

## Outputs

| Stage | Path |
|-------|------|
| Inference | `outputs/generations/add_item_{figstep,mmsafety}/<model>.jsonl` |
| Scoring | `outputs/scores/<model>/<model>.scored.jsonl` |
| Summary | `outputs/scores/<model>/<model>.summary.txt` |

## Dependencies

```bash
pip install -r requirements.txt
```

- CPU: `prepare_base_images.py`, `overlay_math_or_code.py`
- GPU: `batch_qwen_image_edit.py` (CUDA + diffusers)
- API: `meta_infer`, `score_jsonl`

## Citation

```bibtex
@inproceedings{treeteaming2026,
  title={TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration},
  booktitle={CVPR},
  year={2026}
}
```

## License

Research use only. See repository license file. Do not use generated content for harmful purposes.
