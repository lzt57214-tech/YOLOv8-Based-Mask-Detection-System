from __future__ import annotations
"""Validation entry for mini_yolo_app.

中文: mini_yolo_app 的验证入口脚本。
English: Validation entry script for mini_yolo_app.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO, settings as ul_settings

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset.yaml"
DEFAULT_OUTPUT = ROOT / "outputs"
DEFAULT_TRAINED_MODEL = ROOT / "models" / "trained" / "best.pt"


def _resolve(path: str | Path) -> str:
    """Resolve relative paths against project root.

    中文: 将相对路径解析为以 mini_yolo_app 为基准的绝对路径。
    English: Resolve relative paths to absolute paths from mini_yolo_app root.
    """
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return str(p)


def _prepare_runtime() -> None:
    """Keep dataset path behavior consistent with train script.

    中文: 设置 datasets_dir，确保验证时相对数据路径与训练一致。
    English: Set datasets_dir so relative dataset paths in validation match training.
    """
    ul_settings.update({"datasets_dir": str(ROOT)})


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for validation.

    中文: 定义验证命令行参数。
    English: Define command-line arguments for validation.
    """
    parser = argparse.ArgumentParser(description="Validate YOLOv8 model")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--name", default="val")
    return parser


def main() -> None:
    """Run validation with parsed CLI arguments.

    中文: 解析参数并执行验证。
    English: Parse arguments and execute validation.
    """
    args = build_parser().parse_args()
    _prepare_runtime()
    model = YOLO(_resolve(args.model))
    model.val(
        data=_resolve(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=args.workers,
        project=_resolve(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()

