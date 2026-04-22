import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from evaluate_wlasl_model import _load_model


def _best_threshold(scores: np.ndarray, positives: np.ndarray) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.55, 0.0

    candidates = np.unique(np.round(scores, 4))
    candidates = np.concatenate([candidates, np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])])
    candidates = np.unique(np.clip(candidates, 0.05, 0.99))

    best_thr = 0.55
    best_f05 = -1.0
    for thr in candidates:
        pred = scores >= thr
        tp = float(np.sum(pred & positives))
        fp = float(np.sum(pred & ~positives))
        fn = float(np.sum(~pred & positives))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        beta2 = 0.5 * 0.5
        f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)
        if f05 > best_f05:
            best_f05 = f05
            best_thr = float(thr)

    return float(np.clip(best_thr, 0.35, 0.95)), float(best_f05)


def _best_margin(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_thresholds: dict[str, float],
    labels: list[str],
) -> tuple[float, float, float]:
    pred_idx = np.argmax(probs, axis=1)
    top1 = probs[np.arange(len(probs)), pred_idx]
    top2 = np.sort(probs, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    best_margin = 0.12
    best_macro_f1 = -1.0
    best_unknown_rate = 1.0

    for margin in np.linspace(0.0, 0.4, 41):
        adjusted = pred_idx.copy()
        unknown_idx = len(labels)
        for i in range(len(adjusted)):
            label = labels[int(adjusted[i])]
            threshold = class_thresholds.get(label, 0.55)
            if top1[i] < threshold or margins[i] < margin:
                adjusted[i] = unknown_idx

        report = classification_report(
            y_true,
            adjusted,
            labels=list(range(len(labels) + 1)),
            target_names=labels + ["UNKNOWN"],
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = float(report["macro avg"]["f1-score"])
        unknown_rate = float(np.mean(adjusted == unknown_idx))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_unknown_rate = unknown_rate
            best_margin = float(margin)

    return best_margin, best_macro_f1, best_unknown_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize runtime rejection policy")
    parser.add_argument("--dataset", required=True, help="NPZ dataset path")
    parser.add_argument("--model", required=True, help="Model path (.pt or .joblib)")
    parser.add_argument("--output", required=True, help="Output runtime_policy.json path")
    parser.add_argument(
        "--split",
        default="val,test",
        help="Comma-separated split names used for calibration",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.45)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--vote-min-count", type=int, default=3)
    args = parser.parse_args()

    dataset = np.load(Path(args.dataset), allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]
    splits = dataset["splits"].astype(str)

    predict_proba, labels, sequence_length, feature_size, model_type = _load_model(
        Path(args.model),
        X,
    )

    requested_splits = {item.strip() for item in args.split.split(",") if item.strip()}
    mask = np.isin(splits, list(requested_splits))
    if not np.any(mask):
        raise RuntimeError(f"No samples found for split(s): {sorted(requested_splits)}")

    X_eval = X[mask]
    if X_eval.shape[1] != sequence_length or X_eval.shape[2] != feature_size:
        fixed = np.zeros((X_eval.shape[0], sequence_length, feature_size), dtype=np.float32)
        t = min(sequence_length, X_eval.shape[1])
        f = min(feature_size, X_eval.shape[2])
        fixed[:, :t, :f] = X_eval[:, :t, :f]
        X_eval = fixed
    y_eval = y[mask]

    probs = predict_proba(X_eval)

    class_thresholds: dict[str, float] = {}
    class_scores = {}
    for class_idx, label in enumerate(labels):
        scores = probs[:, class_idx]
        positives = y_eval == class_idx
        thr, f05 = _best_threshold(scores, positives)
        class_thresholds[label] = thr
        class_scores[label] = {"threshold": thr, "f0.5": f05}

    default_min_confidence = float(np.median(list(class_thresholds.values())))
    min_margin, macro_f1, unknown_rate = _best_margin(
        probs,
        y_eval,
        class_thresholds,
        labels,
    )

    policy = {
        "model_type": model_type,
        "calibration_splits": sorted(requested_splits),
        "default_min_confidence": float(default_min_confidence),
        "min_margin": float(min_margin),
        "class_thresholds": class_thresholds,
        "ema_alpha": float(args.ema_alpha),
        "vote_window": int(args.vote_window),
        "vote_min_count": int(args.vote_min_count),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    summary = {
        "num_samples": int(X_eval.shape[0]),
        "model_type": model_type,
        "default_min_confidence": default_min_confidence,
        "min_margin": min_margin,
        "macro_f1_with_unknown": macro_f1,
        "unknown_rate": unknown_rate,
    }
    (output_path.parent / "policy_calibration_report.json").write_text(
        json.dumps({"summary": summary, "class_scores": class_scores}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
