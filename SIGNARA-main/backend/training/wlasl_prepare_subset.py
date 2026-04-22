import argparse
import csv
import json
import re
from pathlib import Path


def _load_glosses(glosses_path: Path) -> set[str]:
    with glosses_path.open("r", encoding="utf-8") as handle:
        return {_normalize_gloss(line.strip()) for line in handle if line.strip()}


def _normalize_gloss(gloss: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", gloss.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _safe_get_video_id(instance: dict) -> str | None:
    for key in ("video_id", "video", "id"):
        value = instance.get(key)
        if value:
            return str(value)
    return None


def _safe_get_split(instance: dict) -> str:
    split = str(instance.get("split", "train")).lower()
    if split not in {"train", "val", "test"}:
        return "train"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WLASL subset manifest")
    parser.add_argument("--metadata", required=True, help="Path to WLASL metadata JSON")
    parser.add_argument("--glosses", required=True, help="Path to selected gloss list")
    parser.add_argument("--videos-dir", required=True, help="Directory containing WLASL videos")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    glosses_path = Path(args.glosses)
    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)

    selected_glosses = _load_glosses(glosses_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    rows: list[dict[str, str]] = []
    missing_videos = 0

    for entry in metadata:
        gloss = str(entry.get("gloss", "")).strip()
        if not gloss or _normalize_gloss(gloss) not in selected_glosses:
            continue

        for instance in entry.get("instances", []):
            video_id = _safe_get_video_id(instance)
            if not video_id:
                continue

            candidates = [
                videos_dir / f"{video_id}.mp4",
                videos_dir / f"{video_id}.avi",
                videos_dir / f"{video_id}.mov",
                videos_dir / video_id,
            ]
            video_path = next((path for path in candidates if path.exists()), None)
            if video_path is None:
                missing_videos += 1
                continue

            rows.append(
                {
                    "video_path": str(video_path.resolve()),
                    "gloss": gloss.upper(),
                    "split": _safe_get_split(instance),
                    "video_id": video_id,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "gloss", "split", "video_id"])
        writer.writeheader()
        writer.writerows(rows)

    unique_labels = sorted({row["gloss"] for row in rows})
    print(f"Prepared rows: {len(rows)}")
    print(f"Selected labels present: {len(unique_labels)}")
    print(f"Missing video files skipped: {missing_videos}")


if __name__ == "__main__":
    main()
