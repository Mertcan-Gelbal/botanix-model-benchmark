# Botanix Model Benchmark

A comparative benchmark of deep-learning architectures for large-scale plant **leaf-disease** classification. This repository holds the training/evaluation notebooks and the benchmark protocol; the full Turkish methodology write-up lives in [`notebooks/README.md`](./notebooks/README.md).

**Status:** Research / in progress. Architectures and protocol are defined; measured result artifacts (`results/*.json`) are **not yet committed**, so the comparison tables below are intentionally left blank until runs are published.

---

## Benchmark objective

Answer four questions on a single fixed dataset and protocol:

1. **CNN vs. Transformer** — which family wins on leaf-disease classification?
2. **Segmentation-head effect** — does a self-supervised attention/segmentation head improve classification?
3. **Transfer learning effect** — how much do ImageNet-pretrained weights help vs. training from scratch?
4. **Compute efficiency** — which model is viable for mobile/edge deployment (feeds [Botanix mobile](https://github.com/Mertcan-Gelbal/botanix-mobile-ai))?

## Dataset

Author-published dataset: [Plant Leaf Disease Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/mertcangelbal/plant-leaf-disease-classification-dataset).

| Property | Value (author-reported) |
|---|---|
| Classes | 105 |
| Plant species | 29 |
| Train images | 273,155 |
| Validation images | 50,729 |
| Test images | 50,627 |
| Total | ~374,511 |
| Image size | 256×256 → 224×224 (model input), RGB, JPEG |

> These counts are reported by the dataset author and are consistent across the notebooks, but are not independently re-derived here. **Class-count discrepancy to confirm:** the exported artifacts in [botanix-training](https://github.com/Mertcan-Gelbal/botanix-model-artifacts) list **109** classes (`labels.json`) while this benchmark uses **105**. This is most likely a dataset-version difference. **TODO: confirm and reconcile the two class sets.**
>
> Duplicate/leakage control across the train/val/test split is **not yet documented**. **TODO: document the de-duplication and split procedure.**

## Models

| Notebook | Backbone | Pretrained | Segmentation head | Status | Expected output |
|---|---|---|---|---|---|
| `model1_cnn_baseline.ipynb` | CNN (transfer learning) | ImageNet | No | In progress | `results/cnn_baseline_*` |
| `model2_vit.ipynb` | ViT-B/16 | ImageNet-21k | No | In progress | `results/vit_*` |
| `model3_vit_segmentation.ipynb` | ViT-B/16 | ImageNet-21k | Yes (self-supervised) | Experimental | `results/vit_seg_*` |
| `model4_swin_transformer.ipynb` | Swin-B | ImageNet-22k | No | In progress | `results/swin_*` |
| `model5_swin_segmentation.ipynb` | Swin-B | ImageNet-22k | Yes (self-supervised) | Experimental | `results/swin_seg_*` |
| `botanix_cnn_scratch.ipynb` | Custom CNN (from scratch) | No | No | Experimental | `results/botanix_cnn_scratch_*` |
| `botanix_cnn_scratch_segmentation.ipynb` | Custom CNN (from scratch) | No | Yes | Experimental | `results/botanix_cnn_scratch_seg_*` |
| `botanix_swin_product.ipynb` | Swin (product variant) | ImageNet-22k | Optional | Experimental | product-oriented export |
| `model_comparison.ipynb` | — | — | — | In progress | comparison plots + LaTeX table |

Status legend: **Stable** (finalized, reproducible) · **In progress** (protocol fixed, runs pending/partial) · **Experimental** (idea under evaluation) · **Deprecated** (kept for history).

The segmentation head is **label-free** (self-supervised): it learns a per-patch importance map from classification labels only, regularized by an entropy (sparsity) term, so the model focuses on diseased regions without pixel masks. See [`notebooks/README.md`](./notebooks/README.md) for per-model architecture diagrams.

## Training protocol

| Parameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Epochs | 50 |
| Loss | CrossEntropyLoss (class-weighted) |
| LR schedule | CosineAnnealingLR (η_min=1e-6); +5-epoch warmup for scratch models |
| Grad clip | max_norm=1.0 |
| Normalization | ImageNet mean/std |

Full augmentation tables and per-model architectures are in [`notebooks/README.md`](./notebooks/README.md).

## Results

**Not yet published.** Once the runs complete, `model_comparison.ipynb` emits accuracy / weighted precision / recall / F1, confusion matrices, per-class F1, training time, and inference time per model. This section and the model table's status column will be updated with the measured numbers and the exact evaluation conditions.

## Hardware & reproducibility

- **Hardware used by the author:** single NVIDIA RTX GPU on a Vast.ai instance.
- **Seeds:** set per-notebook; **TODO: pin a single global seed and record it here** for exact reproducibility.
- **Environment:** see [`requirements.txt`](./requirements.txt).

### Setup

```bash
pip install -r requirements.txt

# Configure Kaggle API (do not commit your token)
mkdir -p ~/.kaggle
# place your kaggle.json in ~/.kaggle/ and: chmod 600 ~/.kaggle/kaggle.json

# Download the dataset (each notebook's first cell also does this)
kaggle datasets download -d mertcangelbal/plant-leaf-disease-classification-dataset --unzip -p ./data

# Validate environment and expected paths before running notebooks
python scripts/validate_environment.py
python scripts/check_paths.py

jupyter notebook
```

> Notebooks expect the dataset under `./data` (relative to the notebooks directory). An earlier version hard-coded an absolute local path; use the relative `./data` layout above.

## Repository structure

```
notebooks/           Training & evaluation notebooks + detailed methodology (TR)
scripts/             validate_environment.py, check_paths.py
requirements.txt     Python dependencies
```

## Related repositories

- [Botanix mobile](https://github.com/Mertcan-Gelbal/botanix-mobile-ai) — the on-device app that consumes the winning architecture
- [botanix-training](https://github.com/Mertcan-Gelbal/botanix-model-artifacts) — exported model artifacts

## License

See [LICENSE](./LICENSE) (MIT).
