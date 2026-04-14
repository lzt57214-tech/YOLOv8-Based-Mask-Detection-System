from pathlib import Path
import argparse
import json


def count_labels(labels_dir: Path):
    counts = {
        "train_files": 0,
        "val_files": 0,
        "train_instances": {},
        "val_instances": {},
    }
    for split in ("train", "val"):
        p = labels_dir / split
        files = list(p.glob("*.txt")) if p.exists() else []
        counts[f"{split}_files"] = len(files)
        counts[f"{split}_instances"] = {}
        for f in files:
            text = f.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = parts[0]
                counts[f"{split}_instances"][cls] = counts[f"{split}_instances"].get(cls, 0) + 1

    counts["total_files"] = counts["train_files"] + counts["val_files"]
    # aggregate total_instances
    total_instances = {}
    for d in (counts["train_instances"], counts["val_instances"]):
        for k, v in d.items():
            total_instances[k] = total_instances.get(k, 0) + v
    counts["total_instances"] = total_instances
    return counts


def main():
    parser = argparse.ArgumentParser(description="Count YOLO label instances in train/val folders")
    parser.add_argument("--labels-dir", type=str, default="mark-datas/labels", help="path to labels directory (contains train/ val)")
    parser.add_argument("--out", type=str, default="outputs/labels_count.json", help="output json path")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = count_labels(labels_dir)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(counts, fh, ensure_ascii=False, indent=2)
    print(f"Wrote label counts to {out_path}")


if __name__ == "__main__":
    main()

