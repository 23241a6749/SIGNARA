import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report


def _split_data(X: np.ndarray, y: np.ndarray, splits: np.ndarray):
    train_mask = splits == "train"
    val_mask = np.isin(splits, ["val", "test"])

    if not np.any(train_mask):
        raise RuntimeError("No train samples in dataset")
    if not np.any(val_mask):
        val_mask = ~train_mask

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WLASL baseline sequence model")
    parser.add_argument("--dataset", required=True, help="NPZ from extract_wlasl_landmarks.py")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    splits = data["splits"]
    labels = [str(item) for item in data["labels"].tolist()]

    n_samples, sequence_length, feature_size = X.shape
    X_flat = X.reshape(n_samples, sequence_length * feature_size)

    X_train, y_train, X_val, y_val = _split_data(X_flat, y, splits)

    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(
        y_val,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    artifact = {
        "model": clf,
        "labels": labels,
        "sequence_length": int(sequence_length),
        "feature_size": int(feature_size),
    }
    joblib.dump(artifact, output_dir / "model.joblib")

    metrics = {
        "num_train": int(X_train.shape[0]),
        "num_val": int(X_val.shape[0]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
