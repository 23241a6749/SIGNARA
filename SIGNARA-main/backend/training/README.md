# WLASL v1 Training Pipeline

This pipeline trains a low-latency sign detector using only public WLASL data.

## 1) Prepare subset manifest

```bash
python training/wlasl_prepare_subset.py \
  --metadata /path/to/WLASL_v0.3.json \
  --glosses training/selected_glosses_v1.txt \
  --videos-dir /path/to/wlasl_videos \
  --output training/artifacts/wlasl_v1_manifest.csv
```

## 2) Extract landmark sequences

```bash
python training/extract_wlasl_landmarks.py \
  --manifest training/artifacts/wlasl_v1_manifest.csv \
  --output training/artifacts/wlasl_v1_sequences.npz \
  --sequence-length 32
```

## 3) Train baseline model

```bash
python training/train_wlasl_baseline.py \
  --dataset training/artifacts/wlasl_v1_sequences.npz \
  --output-dir models/wlasl_v1
```

## 3b) Train pose-transformer model (research-backed, recommended)

```bash
python training/train_wlasl_pose_transformer.py \
  --dataset training/artifacts/wlasl_v1_sequences.npz \
  --output-dir models/wlasl_v1 \
  --epochs 30
```

When `models/wlasl_v1/transformer_model.pt` exists, runtime automatically prefers this model over the tree baseline.

## 4) Evaluate model quality

```bash
python training/evaluate_wlasl_model.py \
  --dataset training/artifacts/wlasl_v1_sequences.npz \
  --model models/wlasl_v1/transformer_model.pt \
  --output-dir models/wlasl_v1/eval
```

You can also evaluate the tree baseline by passing `models/wlasl_v1/model.joblib`.

## 5) Calibrate runtime rejection policy (automatic)

```bash
python training/optimize_runtime_policy.py \
  --dataset training/artifacts/wlasl_v1_sequences.npz \
  --model models/wlasl_v1/transformer_model.pt \
  --output models/wlasl_v1/runtime_policy.json
```

Then re-run evaluation with policy enabled:

```bash
python training/evaluate_wlasl_model.py \
  --dataset training/artifacts/wlasl_v1_sequences.npz \
  --model models/wlasl_v1/transformer_model.pt \
  --policy models/wlasl_v1/runtime_policy.json \
  --output-dir models/wlasl_v1/eval_with_policy
```

## One-command full pipeline

```bash
python training/run_pipeline.py \
  --metadata /path/to/WLASL_v0.3.json \
  --videos-dir /path/to/wlasl_videos
```

This executes: prepare subset -> extract landmarks -> train -> evaluate -> optimize policy -> evaluate with policy.

Artifacts expected by runtime service:

- `models/wlasl_v1/model.joblib`
- `models/wlasl_v1/metrics.json`

## Notes

- This v1 model is optimized for a small vocabulary (12 glosses by default).
- Latency and stability are improved via sequence buffering, EMA smoothing, confidence thresholding, and margin-based rejection (`UNKNOWN`).
- No personal webcam data is required for training.

## Exactly when to evaluate and test

1. **After extraction**: quick sanity check (class counts, split sizes) from extraction logs.
2. **After each training run**: run step 4 (`evaluate_wlasl_model.py`) and compare `accuracy_top1`, `macro_f1`.
3. **Before runtime testing**: run step 5 (`optimize_runtime_policy.py`) and then evaluate with policy.
4. **Only then run live test** (`uvicorn` + frontend) so `runtime_policy.json` is applied.

Recommended acceptance gates before live testing:

- `accuracy_top1 >= 0.80` on val/test subset
- `macro_f1 >= 0.75`
- `unknown_rate` in policy eval between `0.05` and `0.35` (too low = noisy labels, too high = over-rejection)

## Reputable-paper alignment

- **MediaPipe Hands (CVPR Workshop 2020)** for real-time landmark extraction.
- **SAM-SLR (CVPR Workshop 2021)** and **ST-GCN (AAAI 2018)** motivate skeleton-temporal modeling.
- **SignGraph (CVPR 2024)** and **Uni-Sign (ICLR 2025)** motivate graph/transformer-style sequence modeling.
- Current v1 implementation follows this by using pose sequences + lightweight transformer with strict real-time constraints.
