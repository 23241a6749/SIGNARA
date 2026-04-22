import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import torch
    from src.model.pose_transformer import PoseTransformerClassifier
except Exception:
    torch = None
    PoseTransformerClassifier = None


def _topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    hits = [target in row for target, row in zip(targets, topk)]
    return float(np.mean(hits)) if hits else 0.0


def _write_confusion_csv(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])


def _load_model(model_path: Path, X: np.ndarray):
    suffix = model_path.suffix.lower()
    if suffix == ".joblib":
        artifact = joblib.load(model_path)
        model = artifact["model"]
        labels = [str(label).upper() for label in artifact["labels"]]
        sequence_length = int(artifact.get("sequence_length", X.shape[1]))
        feature_size = int(artifact.get("feature_size", X.shape[2]))

        def predict_proba(x_sequences: np.ndarray) -> np.ndarray:
            flattened = x_sequences.reshape(-1, sequence_length * feature_size)
            return model.predict_proba(flattened)

        model_type = "sklearn_trees"
        return predict_proba, labels, sequence_length, feature_size, model_type

    if suffix == ".pt":
        if torch is None or PoseTransformerClassifier is None:
            raise RuntimeError("Torch-based model requested but torch is unavailable")

        checkpoint = torch.load(model_path, map_location="cpu")
        labels = [str(label).upper() for label in checkpoint["labels"]]
        sequence_length = int(checkpoint.get("sequence_length", X.shape[1]))
        feature_size = int(checkpoint.get("feature_size", X.shape[2]))
        hidden_dim = int(checkpoint.get("hidden_dim", 256))
        num_heads = int(checkpoint.get("num_heads", 4))
        num_layers = int(checkpoint.get("num_layers", 3))
        dropout = float(checkpoint.get("dropout", 0.2))

        model = PoseTransformerClassifier(
            input_dim=feature_size,
            num_classes=len(labels),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        def predict_proba(x_sequences: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.from_numpy(x_sequences.astype(np.float32))
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs

        model_type = "pose_transformer"
        return predict_proba, labels, sequence_length, feature_size, model_type

    raise RuntimeError(f"Unsupported model format: {model_path}")


def _apply_policy(
    probs: np.ndarray,
    labels: list[str],
    policy: Dict,
) -> np.ndarray:
    default_min_conf = float(policy.get("default_min_confidence", 0.0))
    min_margin = float(policy.get("min_margin", 0.0))
    class_thresholds = {
        str(k).upper(): float(v) for k, v in policy.get("class_thresholds", {}).items()
    }

    preds = np.argmax(probs, axis=1)
    scored = probs[np.arange(len(probs)), preds]
    top2 = np.sort(probs, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]

    adjusted = preds.copy()
    unknown_idx = len(labels)
    for i in range(len(adjusted)):
        label = labels[int(adjusted[i])]
        threshold = class_thresholds.get(label, default_min_conf)
        if scored[i] < threshold or margins[i] < min_margin:
            adjusted[i] = unknown_idx

    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained WLASL baseline model")
    parser.add_argument("--dataset", required=True, help="NPZ dataset path")
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory")
    parser.add_argument(
        "--split",
        default="val,test",
        help="Comma-separated split names to evaluate (default: val,test)",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Optional runtime_policy.json for rejection-aware evaluation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(Path(args.dataset), allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]
    splits = dataset["splits"].astype(str)

    predict_proba, labels, sequence_length, feature_size, model_type = _load_model(
        Path(args.model),
        X,
    )

    requested_splits = {item.strip() for item in args.split.split(",") if item.strip()}
    eval_mask = np.isin(splits, list(requested_splits))
    if not np.any(eval_mask):
        raise RuntimeError(f"No samples found for split(s): {sorted(requested_splits)}")

    X_eval = X[eval_mask]
    if X_eval.shape[1] != sequence_length or X_eval.shape[2] != feature_size:
        X_fixed = np.zeros((X_eval.shape[0], sequence_length, feature_size), dtype=np.float32)
        t = min(sequence_length, X_eval.shape[1])
        f = min(feature_size, X_eval.shape[2])
        X_fixed[:, :t, :f] = X_eval[:, :t, :f]
        X_eval = X_fixed

    y_eval = y[eval_mask]

    probs = predict_proba(X_eval)
    preds = np.argmax(probs, axis=1)
    eval_labels = labels.copy()

    if args.policy:
        policy_path = Path(args.policy)
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        preds = _apply_policy(probs, labels, policy)
        eval_labels = labels + ["UNKNOWN"]

    report = classification_report(
        y_eval,
        preds,
        labels=list(range(len(eval_labels))),
        target_names=eval_labels,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_eval, preds, labels=list(range(len(eval_labels))))

    unknown_rate = 0.0
    if "UNKNOWN" in eval_labels:
        unknown_idx = eval_labels.index("UNKNOWN")
        unknown_rate = float(np.mean(preds == unknown_idx))

    metrics = {
        "num_samples": int(X_eval.shape[0]),
        "splits": sorted(requested_splits),
        "model_type": model_type,
        "accuracy_top1": float(report["accuracy"]),
        "accuracy_top3": _topk_accuracy(probs, y_eval, 3),
        "accuracy_top5": _topk_accuracy(probs, y_eval, 5),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "unknown_rate": unknown_rate,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_confusion_csv(cm, labels, output_dir / "confusion_matrix.csv")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
