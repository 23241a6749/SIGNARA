import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(step_name: str, command: list[str], workdir: Path) -> None:
    printable = " ".join(command)
    print(f"\n[PIPELINE] {step_name}")
    print(f"[PIPELINE] cmd: {printable}")
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    workdir_path = str(workdir)
    env["PYTHONPATH"] = (
        f"{workdir_path}{os.pathsep}{current}" if current else workdir_path
    )
    subprocess.run(command, cwd=str(workdir), check=True, env=env)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end WLASL training/evaluation/policy pipeline"
    )
    parser.add_argument("--metadata", required=True, help="WLASL metadata JSON path")
    parser.add_argument("--videos-dir", required=True, help="WLASL videos directory")
    parser.add_argument(
        "--glosses",
        default="training/selected_glosses_v1.txt",
        help="Gloss list file",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="training/artifacts",
        help="Intermediate artifacts directory",
    )
    parser.add_argument(
        "--model-dir",
        default="models/wlasl_v1",
        help="Model output directory",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means use all samples",
    )
    parser.add_argument(
        "--split",
        default="val,test",
        help="Evaluation/calibration split names",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip transformer training and run tree baseline only",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip subset preparation step",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip landmark extraction step",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = (backend_dir / args.artifacts_dir).resolve()
    model_dir = (backend_dir / args.model_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts_dir / "wlasl_v1_manifest.csv"
    sequences_path = artifacts_dir / "wlasl_v1_sequences.npz"
    policy_path = model_dir / "runtime_policy.json"

    py = sys.executable

    if not args.skip_prepare:
        cmd = [
            py,
            "training/wlasl_prepare_subset.py",
            "--metadata",
            str(Path(args.metadata).resolve()),
            "--glosses",
            str((backend_dir / args.glosses).resolve()),
            "--videos-dir",
            str(Path(args.videos_dir).resolve()),
            "--output",
            str(manifest_path),
        ]
        _run("Prepare subset manifest", cmd, backend_dir)

    if not args.skip_extract:
        cmd = [
            py,
            "training/extract_wlasl_landmarks.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(sequences_path),
            "--sequence-length",
            str(args.sequence_length),
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        _run("Extract landmarks", cmd, backend_dir)

    baseline_model = model_dir / "model.joblib"
    transformer_model = model_dir / "transformer_model.pt"

    if args.baseline_only:
        _run(
            "Train baseline tree model",
            [
                py,
                "training/train_wlasl_baseline.py",
                "--dataset",
                str(sequences_path),
                "--output-dir",
                str(model_dir),
            ],
            backend_dir,
        )
        target_model = baseline_model
    else:
        _run(
            "Train pose transformer",
            [
                py,
                "training/train_wlasl_pose_transformer.py",
                "--dataset",
                str(sequences_path),
                "--output-dir",
                str(model_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ],
            backend_dir,
        )
        target_model = transformer_model if transformer_model.exists() else baseline_model

    eval_dir = model_dir / "eval"
    _run(
        "Evaluate model",
        [
            py,
            "training/evaluate_wlasl_model.py",
            "--dataset",
            str(sequences_path),
            "--model",
            str(target_model),
            "--split",
            args.split,
            "--output-dir",
            str(eval_dir),
        ],
        backend_dir,
    )

    _run(
        "Optimize runtime policy",
        [
            py,
            "training/optimize_runtime_policy.py",
            "--dataset",
            str(sequences_path),
            "--model",
            str(target_model),
            "--split",
            args.split,
            "--output",
            str(policy_path),
        ],
        backend_dir,
    )

    eval_policy_dir = model_dir / "eval_with_policy"
    _run(
        "Evaluate model with runtime policy",
        [
            py,
            "training/evaluate_wlasl_model.py",
            "--dataset",
            str(sequences_path),
            "--model",
            str(target_model),
            "--split",
            args.split,
            "--policy",
            str(policy_path),
            "--output-dir",
            str(eval_policy_dir),
        ],
        backend_dir,
    )

    summary = {
        "model_path": str(target_model),
        "policy_path": str(policy_path),
        "eval": _load_json(eval_dir / "metrics.json"),
        "eval_with_policy": _load_json(eval_policy_dir / "metrics.json"),
    }
    summary_path = model_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[PIPELINE] Completed successfully")
    print(f"[PIPELINE] Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
