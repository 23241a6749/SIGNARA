# Sign Detection Research Notes (WLASL-first)

## Sources reviewed

- MediaPipe Hands (`arXiv:2006.10214`, CVPR Workshop 2020) for real-time landmark extraction.
- WLASL benchmark (`arXiv:1910.11006`, WACV 2020 accepted) for ASL word-level scaling.
- ST-GCN (`arXiv:1801.07455`, AAAI 2018) for spatial-temporal skeleton modeling.
- SAM-SLR (`arXiv:2103.08833`, CVPR Workshop 2021) for strong skeleton-aware isolated SLR design.
- SignGraph (CVPR 2024 OpenAccess) for graph-based sign sequence representation.
- Uni-Sign (`arXiv:2501.15187`, ICLR 2025 accepted) for modern unified pose/RGB understanding.
- Sign Pose-based Transformer (`maty-bohacek/spoter`, WACV 2022 Workshop) for low-compute pose transformer direction.
- MediaPipe + DTW practical repo (`gabguerin/Sign-Language-Recognition--MediaPipe-DTW`) for thresholded voting ideas.
- MediaPipe + lightweight landmark classifier repos (`Kazuhito00/...`, `kinivi/...`) for deployment-oriented baseline design.

## Why this implementation path

1. Landmark-based models are significantly lower latency than RGB video models for small vocabularies.
2. WLASL subset training avoids personal data collection and still gives signer-diverse training samples.
3. Reputable recent work (SignGraph/Uni-Sign) supports temporal sequence modeling over pure frame-wise classification.
4. Sequence buffering + EMA + rejection thresholds reduce unstable live predictions.
5. Lightweight transformer offers a better accuracy/latency tradeoff than large multimodal models for this v1 scope.

## Future upgrade path

- Move from small-vocab isolated signs to larger vocab and continuous SLR with CTC-style decoding.
- Export to ONNX Runtime for tighter latency budgets.
- Add signer-adaptive calibration (without full retraining) if domain shifts are observed.
