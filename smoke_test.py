"""Dataset path smoke test.

中文: 数据集路径冒烟测试脚本。
English: Smoke test utility for dataset path validation.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset.yaml"


if __name__ == "__main__":
    # 中文: 读取数据配置并检查 train/val 目录是否存在。
    # English: Load dataset config and verify train/val directories exist.
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    data_root = (DATA_YAML.parent / data["path"]).resolve()

    train_dir = data_root / "images" / "train"
    val_dir = data_root / "images" / "val"

    assert train_dir.exists(), f"Missing train dir: {train_dir}"
    assert val_dir.exists(), f"Missing val dir: {val_dir}"

    print("smoke_test passed")
    print(f"dataset root: {data_root}")
    print(f"train dir: {train_dir}")
    print(f"val dir: {val_dir}")

