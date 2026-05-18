# MedXAI: Explainable Medical Imaging Intelligence System

**Classify chest and brain scans with confidence — and show clinicians *why* the model decided.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Medical imaging drives a large share of diagnostic workflows, yet deep learning models often behave as opaque scorers: high accuracy on benchmarks does not translate into trust at the bedside. Radiologists need more than a label—they need spatial evidence aligned with anatomy and protocol.

**MedXAI** addresses that gap by pairing multi-architecture classification (normal, benign tumor, malignant tumor) with model-native interpretability: Grad-CAM for convolutional backbones and attention rollout for Vision Transformers. The system is designed as a deployable research platform—train, serve, and review explanations in one pipeline — without substituting for clinical judgment or regulatory clearance.

---

## System Overview

End-to-end flow from curated DICOM/NIfTI or PNG cohorts through training, checkpointing, REST inference, and an interactive review UI.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA & PREPROCESSING │
│ Raw imaging (X-ray / MRI) → resize, normalize, augment → train/val/test │
└───────────────────────────────────┬─────────────────────────────────────────┘
 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL ZOO (PyTorch) │
│ ResNet-50 │ EfficientNet-B3 │ ViT-B/16 │ optional weighted Ensemble │
└───────────────────────────────────┬─────────────────────────────────────────┘
 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPLAINABILITY ENGINE │
│ CNN: Grad-CAM (+ guided optional) │ ViT: Attention Rollout / last-layer │
└───────────────────────────────────┬─────────────────────────────────────────┘
 ▼
┌──────────────────────────────┐ ┌──────────────────────────────────────────┐
│ FastAPI inference service │◄──►│ Streamlit clinician / researcher UI │
│ /predict /explain /health │ │ upload → class probs → heatmap overlay │
└──────────────────────────────┘ └──────────────────────────────────────────┘
```

**Three layers at a glance**

| Layer | Focus |
|-------|--------|
| **Product** | Upload a study slice, receive class probabilities and an overlaid saliency map in seconds. |
| **Engineering** | Modular trainers, shared preprocessing, versioned checkpoints, OpenAPI-backed API, container-ready layout. |
| **Research** | Comparable CNN vs ViT baselines, reproducible splits, logged metrics, and exportable explanation artifacts for qualitative review. |

---

## Key Features

- **Multi-model classification** — ResNet-50, EfficientNet-B3, ViT-B/16, and optional softmax-weighted ensemble with per-model calibration hooks.
- **Explainability engine** — Grad-CAM for CNNs; attention rollout and last-block attention maps for ViT; unified overlay export (PNG, numpy).
- **Training pipeline** — Stratified k-fold or hold-out splits, mixed-precision optional, TensorBoard / CSV logging, early stopping on macro-F1.
- **Inference API** — FastAPI service with batch and single-image routes, structured JSON for classes and explanation URIs or base64 payloads.
- **Review UI** — Streamlit dashboard for side-by-side image, probability bar chart, and adjustable heatmap opacity.
- **Reproducibility** — Fixed seeds, config-driven experiments (`configs/`), and experiment IDs tied to checkpoint directories.

---

## Models

Three-class head: `normal` · `benign` · `malignant`. Image input default: **224×224** (3-channel, modality-specific normalization in config).

| Model | Backbone | Params (approx.) | Typical val macro-F1* | Explainability |
|-------|----------|------------------|------------------------|----------------|
| **ResNet-50** | ImageNet-pretrained CNN | 25M | 0.84–0.88 | Grad-CAM on final conv block |
| **EfficientNet-B3** | Compound-scaled CNN | 12M | 0.86–0.90 | Grad-CAM on `_conv_head` |
| **ViT-B/16** | Patch transformer | 86M | 0.85–0.89 | Attention rollout, CLS-to-patch |
| **Ensemble** | Softmax vote / learned weights | — | +1–3 pts vs best single* | Per-member maps + fused overlay |

\*Illustrative ranges on internal hold-out; **your dataset and protocol will differ.** Always re-validate before any downstream use.

Checkpoints live under `checkpoints/{model_name}/best.pt`. Register the active model in `configs/inference.yaml`.

---

## Explainability

### Grad-CAM (CNN)

Target the last convolutional stage before global pooling. For predicted class \(c\):

1. Forward pass with gradients enabled on the target layer.
2. Global-average-pool gradient weights per channel.
3. Weighted sum of feature maps → ReLU → upsample to input resolution.
4. Normalize and alpha-blend with the source image (configurable colormap: `jet`, `inferno`).

Optional **Guided Grad-CAM** masks negative gradients for sharper boundaries (slower, research use).

### ViT attention

- **Last-layer attention** — CLS token attention to patch tokens, reshaped to 2D grid.
- **Attention rollout** — Accumulate attention across layers with residual reweighting for a single global relevance map.

Artifacts are written to `outputs/explanations/{request_id}/` alongside metadata JSON (class index, layer name, timestamp).

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for training (CPU supported for smoke tests)
- 8 GB+ RAM; 16 GB+ for ViT batch training

### Install

```bash
git clone https://github.com/meolen07/explainable-medical-imaging-ai.git
cd explainable-medical-imaging-ai
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Prepare data

Place data in the layout expected by `configs/data.yaml` (example):

```
data/
├── train/
│ ├── normal/
│ ├── benign/
│ └── malignant/
├── val/
└── test/
```

Update paths and modality-specific normalization in `configs/data.yaml`.

### Train

```bash
# Single model
python -m src.train --config configs/train_resnet50.yaml

# EfficientNet
python -m src.train --config configs/train_efficientnet_b3.yaml

# ViT
python -m src.train --config configs/train_vit_b16.yaml

# Ensemble (after member checkpoints exist)
python -m src.train --config configs/train_ensemble.yaml
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Inference (CLI)

```bash
python -m src.predict \
 --checkpoint checkpoints/resnet50/best.pt \
 --image path/to/scan.png \
 --explain \
 --output outputs/predictions/
```

### API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@path/to/scan.png" \
 -F "model=resnet50" \
 -F "explain=true"
```

OpenAPI docs: `http://localhost:8000/docs`

### UI

```bash
streamlit run src/ui/app.py
```

Set `API_BASE_URL` in `.env` if the UI talks to a remote FastAPI instance.

### Docker (optional)

```bash
docker compose up --build
```

---

## Project Structure

```
explainable-medical-imaging-ai/
├── configs/ # YAML: data, train, inference per model
├── data/ # Local dataset (gitignored); see data/README.md
├── checkpoints/ # Saved weights (gitignored)
├── runs/ # TensorBoard logs
├── outputs/ # Predictions & explanation overlays
├── src/
│ ├── data/ # datasets, transforms, splits
│ ├── models/ # ResNet, EfficientNet, ViT, ensemble builders
│ ├── explain/ # grad_cam.py, attention_rollout.py, overlays
│ ├── train.py # training entrypoint
│ ├── predict.py # CLI inference + explain export
│ ├── api/
│ │ └── main.py # FastAPI app
│ └── ui/
│ └── app.py # Streamlit reviewer
├── tests/ # unit tests (metrics, explain shapes)
├── scripts/ # download weights, export ONNX (optional)
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Limitations & Disclaimer

This repository is a **research and engineering reference**. It is **not** a medical device, **not** FDA/CE-marked, and **not** validated for clinical diagnosis, treatment, or triage.

- Models may fail on out-of-distribution scanners, contrast protocols, pediatrics, or rare pathologies.
- Explainability maps are **heuristic attributions**, not ground-truth lesion segmentation; they can highlight spurious correlates.
- Reported metrics are dataset-specific; external validation is mandatory before any real-world consideration.
- **Do not use outputs to make patient care decisions.** Qualified clinicians remain the authority on interpretation.

Use only on de-identified data compliant with your institution’s IRB, HIPAA, GDPR, or equivalent policies.

---

## Key Insight

**Accuracy and interpretability are jointly optimized, not traded off by default.**

A model that is correct for the wrong reasons is a liability in medicine. MedXAI treats explanation generation as a first-class output—same request lifecycle as the prediction—so teams can audit failure modes, compare architectures (CNN locality vs ViT global context), and document model behavior for publications and internal review. The goal is not to replace the radiologist’s report, but to make model behavior **inspectable** before anyone asks it to matter.

---

## Citation

If you use this work in academic projects, please cite:

```bibtex
@software{medxai2025,
 author = {Nguyen, Huynh Mai Linh},
 title = {MedXAI: Explainable Medical Imaging Intelligence System},
 year = {2025},
 publisher = {GitHub},
 url = {https://github.com/meolen07/explainable-medical-imaging-ai},
 note = {Research software; not for clinical use.}
}
```

---

## Author

**Huynh Mai Linh Nguyen**  
Research Engineer · Medical AI  

- GitHub: [@meolen07](https://github.com/meolen07)  
- Repository: [MedXAI / explainable-medical-imaging-ai](https://github.com/meolen07/explainable-medical-imaging-ai)

Contributions welcome via pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for style and test expectations.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

<p align="center">
 <sub>MedXAI — built for transparent experimentation, not clinical deployment.</sub>
</p>
