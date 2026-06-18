# QCAgent: An agentic framework for quality-controllable pathology report generation from whole slide image

qcagent is a modular pipeline for pathology report quality control from whole-slide images (WSIs). It combines cross-modal patch retrieval, patch-level visual reasoning, and iterative large-language-model refinement.

This repository is the open-source release package. Large datasets, model weights, intermediate results, and logs are intentionally excluded.

## Framework Overview

<div align="center">
  <img src="./img/Pipeline.png" alt="QCAgent framework overview" width="100%" />
</div>


## Prerequisites (Important for Reproduction)

This repository contains the orchestration code only. Before running, you must
prepare the following external components and pre-computed inputs:

- External model repositories / weights (set via env vars below):
  - CONCH (text/image encoder for retrieval)
  - Patho-R1 (patch-level pathology VLM)
  - PRISM (WSI-level foundation model) to generate the initial draft reports
- Pre-computed inputs placed under `data/` (see Expected Data Layout):
  - CONCH patch features (`.h5`) and patch coordinates (`.h5`)
  - PRISM-generated WSI-level draft reports (one `.txt` per slide)
  - Original WSI slides (`.svs`) for high-resolution patch cropping

Note: feature extraction (CONCH) and the PRISM draft generation are not part of
this repository; use the official upstream tools to produce them.

## Repository Structure

```text
.
|-- run_qc.py                    # Main CLI entry for the QC pipeline
|-- pipeline.py                  # Pipeline orchestration
|-- retrieval/                   # CONCH-based patch retrieval
|-- inference/                   # Patho-R1 patch-level report generation
|-- qc_agent/                    # LLM interaction, prompting, parsing, QC logic
|-- img/Pipeline.png             # Framework overview figure
|-- evaluate.py                  # Core evaluation metrics
|-- run_patho_r1_inference.py    # Standalone Patho-R1 patch inference utility
|-- data/README.md               # Data layout instructions (no data included)
|-- requirements.txt
|-- .env.example
`-- LICENSE
```


## Installation

```bash
conda activate your_env
pip install -r requirements.txt
```

## Environment Variables

Minimum required for Qwen backend:

```bash
export QWEN_API_KEY="your_qwen_api_key"
```

Recommended full setup:

```bash
export QWEN_API_KEY="your_qwen_api_key"
export CONCH_REPO_PATH="/path/to/CONCH"
export PATHO_R1_REPO_PATH="/path/to/Patho-R1"
export PATHO_R1_MODEL_PATH="/path/to/Patho-R1/models/Patho-R1-7B"

# Optional Dify backend
export DIFY_API_KEY="your_dify_api_key"
export DIFY_BASE_URL="http://localhost/v1"
```

The Qwen backend calls an OpenAI-compatible chat endpoint. The default base URL
is a third-party provider; override it for your own provider with
`--qwen_base_url` (or the corresponding client setting). Make sure the
`--qwen_model` string matches the model name exposed by your endpoint.

You can also copy values from .env.example into your local environment configuration.

## Expected Data Layout

Prepare your local data directory as follows:

```text
data/
  reports_batch_TCGA/      # PRISM WSI-level draft reports: <slide_id>.txt
  tcga_stad_reports/       # (optional) cached patch-level reports
  features_conch_v1/       # CONCH patch features: <slide_id>.h5
  patches/                 # patch coordinates: <slide_id>_patches.h5
  TCGA-STAD/          # original WSIs: <slide_id>.svs
  tcga_reports_cleaned.csv # evaluation ground truth (slide_id, slide_reports)
```

For TCGA-STAD, the evaluation ground-truth reports are taken from TITAN.
See data/README.md for notes and constraints.

## Quick Start

Single-slide run (requires the PRISM draft report and CONCH features for the slide):

```bash
python run_qc.py \
  --slide_id TCGA-XXXX \
  --backend qwen \
  --qwen_model "Qwen/Qwen3-VL-30B-A3B-Thinking" \
  --max_rounds 3 \
  --topk 3 \
  --wsi_report_dir ./data/reports_batch_TCGA \
  --features_dir ./data/features_conch_v1 \
  --coords_dir ./data/patches \
  --svs_dir ./data/TCGA-STAD \
  --output_dir ./qc_results
```

Batch run:

```bash
python run_qc.py \
  --batch \
  --slide_list ./data/slide_list_ready.txt \
  --backend qwen \
  --output_dir ./qc_results
```

## Evaluation

Evaluate generated reports:

```bash
python evaluate.py \
  --gt_csv ./data/tcga_reports_cleaned.csv \
  --results_dir ./qc_results \
  --output ./eval_results.json
```

## Reproducibility Notes

- This repository does not redistribute third-party model code or model weights.
- Please ensure your use of external models and datasets complies with their licenses and institutional policy.
- For fair comparison, keep dataset split definitions and preprocessing consistent across experiments.

## License

This project is released under the MIT License. See LICENSE.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{wang2026qcagent,
  title={QCAgent: An agentic framework for quality-controllable pathology report generation from whole slide image},
  author={Wang, Rundong and Ba, Wei and Zhou, Ying and Li, Yingtai and Liu, Bowen and Wang, Baizhi and Wang, Yuhao and Yang, Zhidong and Zhang, Kun and Yan, Rui and others},
  journal={arXiv preprint arXiv:2603.01647},
  year={2026}
}
```
