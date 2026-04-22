import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model.pose_transformer import PoseTransformerClassifier


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sequence = self.X[idx].copy()
        label = self.y[idx]

        if self.augment:
            if np.random.rand() < 0.3:
                noise = np.random.normal(0.0, 0.01, size=sequence.shape)
                sequence += noise.astype(np.float32)

            if np.random.rand() < 0.2:
                drop_idx = np.random.randint(0, sequence.shape[0])
                sequence[drop_idx] = sequence[max(0, drop_idx - 1)]

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)


def _split_data(X: np.ndarray, y: np.ndarray, splits: np.ndarray):
    train_mask = splits == "train"
    val_mask = np.isin(splits, ["val", "test"])

    if not np.any(train_mask):
        raise RuntimeError("No train samples in dataset")
    if not np.any(val_mask):
        val_mask = ~train_mask

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    losses = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            pred1 = probs.argmax(dim=-1)
            top1 += (pred1 == yb).sum().item()

            topk = torch.topk(probs, k=min(3, probs.size(-1)), dim=-1).indices
            top3 += sum(int(yb[i].item() in topk[i].tolist()) for i in range(len(yb)))
            total += len(yb)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": float(top1 / total) if total else 0.0,
        "top3": float(top3 / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight pose transformer on WLASL")
    parser.add_argument("--dataset", required=True, help="NPZ from extract_wlasl_landmarks.py")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(Path(args.dataset), allow_pickle=True)
    X = data["X"]
    y = data["y"]
    splits = data["splits"].astype(str)
    labels = [str(item).upper() for item in data["labels"].tolist()]

    X_train, y_train, X_val, y_val = _split_data(X, y, splits)
    num_classes = len(labels)
    sequence_length = X.shape[1]
    feature_size = X.shape[2]

    train_ds = SequenceDataset(X_train, y_train, augment=True)
    val_ds = SequenceDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseTransformerClassifier(
        input_dim=feature_size,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        train_loss = float(np.mean(losses)) if losses else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    checkpoint = {
        "state_dict": best_state,
        "labels": labels,
        "sequence_length": sequence_length,
        "feature_size": feature_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    torch.save(checkpoint, output_dir / "transformer_model.pt")

    metrics = {
        "num_train": int(X_train.shape[0]),
        "num_val": int(X_val.shape[0]),
        "best_val_top1": float(best_top1),
        "best_val_top3": float(max(item["val_top3"] for item in history)),
        "device": str(device),
    }
    (output_dir / "transformer_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transformer_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
