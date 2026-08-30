# DuaLoc: Dual-Encoder Bug Localization

Replication package for **DuaLoc**, a bug-localization framework that pairs
**UniXcoder** (semantic encoder) and **GraphCodeBERT** (structural encoder),
each independently fine-tuned with an **InfoNCE contrastive objective** on
(bug report, function) pairs using in-batch negatives. A
**bug-report-conditioned attention** mechanism aggregates function embeddings
into query-dependent file representations; the resulting cosine-similarity
scores (`σ_unixcoder_summary`, `σ_unixcoder_description`,
`σ_graphcodebert_summary`, `σ_graphcodebert_description`) are fused with four
classical IR features — surface lexical similarity (**φ1**), collaborative
filtering (**φ2**), bug-fixing recency (**φ3**), and bug-fixing frequency
(**φ4**) — into an 8-dim vector scored by a pointwise linear ranker (Huber
loss + L1 regularization, trained with SGD). The pipeline is evaluated with a
consecutive-fold protocol on **Top-1/5/10, MAP, and MRR** across six
open-source Java projects (Eclipse Platform UI, JDT, BIRT, SWT, Tomcat,
AspectJ).

## 1. Requirements

- **Python 3.8+** (3.10 recommended).
- Dependencies (see `requirements.txt`): `torch`, `transformers`, `numpy`,
  `scipy`, `nltk`, `tree-sitter`, `tree-sitter-java`.
  - `nltk` stopwords are auto-downloaded on first run (`preprocess.py`).
  - `tree-sitter`/`tree-sitter-java` are optional — if unavailable,
    `preprocess.py` falls back to a regex-based Java function extractor.
- **Hardware**: the paper's experiments ran on a single NVIDIA RTX 5090.
  `--device` defaults to `cuda` in `run_pipeline.py` / `finetune_contrastive.py`,
  but `encoders.py` auto-falls back to CPU if CUDA is unavailable .

## 2. Setup

```bash
git clone https://github.com/amanyfaisal11/buglocalization.git
cd buglocalization
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## 3. Dataset layout

`scripts/parse_dataset.py`  run from the repo root:

```
dataset/AspectJ.xml
dataset/Birt.xml
dataset/Eclipse_Platform_UI.xml
dataset/JDT.xml
dataset/SWT.xml
dataset/Tomcat.xml
```

```bash
python scripts/parse_dataset.py
```

Dataset download:
https://figshare.com/articles/dataset/The_dataset_of_six_open_source_Java_projects/951967

## 4. Running the pipeline

Full pipeline, one project at a time:

```bash
python run_pipeline.py \
  --project eclipse_platform_ui \
  --bug_reports eclipse_platform_ui_bugs.json \
  --source_files /path/to/eclipse_platform_ui_source \
  --output_dir output/eclipse_platform_ui \
  --device cuda \
  --contrastive_epochs 5
```

This runs, in order: preprocessing + ground-truth resolution → hard-negative
mining (top-200 by φ1) → IR feature extraction (φ1–φ4) → independent
contrastive fine-tuning of UniXcoder and GraphCodeBERT → attention-based
embedding caching → per-fold LTR training/scoring over the project's
consecutive train/test fold pairs → averaged metrics.

Individual stages also have standalone CLIs (`preprocess.py`, `fold_split.py`,
`train_ltr.py`, and `evaluate.py` are library-only and have no `__main__`):

```bash
python mine_hard_negatives.py --bug_reports <bugs.json> --source_files <src> \
  --output output/hard_negatives.json --top_k 200

python extract_ir_features.py --bug_reports <bugs.json> --source_files <src> \
  --hard_negatives output/hard_negatives.json --output output/ir_features.pkl

python finetune_contrastive.py --bug_reports <bugs.json> --source_files <src> \
  --output_dir output --device cuda --epochs 5 --batch_size 64
```
