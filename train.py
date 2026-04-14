from __future__ import annotations
"""Train entry for mini_yolo_app.

中文: mini_yolo_app 的训练入口脚本。
English: Training entry script for mini_yolo_app.
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO, settings as ul_settings

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset.yaml"
DEFAULT_OUTPUT = ROOT / "outputs"
DEFAULT_PRETRAINED_MODEL = ROOT / "models" / "pretrained" / "yolov8n.pt"
DEFAULT_TRAINED_MODEL = ROOT / "models" / "trained" / "best.pt"


def _resolve(path: str | Path) -> str:
    """Resolve relative paths against project root.

    中文: 将相对路径转换为以 mini_yolo_app 为基准的绝对路径。
    English: Convert relative paths to absolute paths based on mini_yolo_app.
    """
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return str(p)


def _prepare_runtime() -> None:
    """Set runtime defaults for portable dataset config.

    中文: 将 Ultralytics 的 datasets_dir 指向当前项目目录，
    使 dataset.yaml 中 `path: mark-datas` 这类相对路径在不同机器上都能正确解析。
    English: Point Ultralytics datasets_dir to this project root so relative dataset
    paths such as `path: mark-datas` resolve consistently across machines.
    """
    ul_settings.update({"datasets_dir": str(ROOT)})


def _prepare_data_yaml(data_yaml: str) -> str:
    """Create runtime dataset yaml with absolute root path.

    中文: 生成运行时数据集配置，将相对 path 转换为绝对路径，避免 Ultralytics
    在某些环境中把相对路径重定向到默认 datasets 目录。
    English: Materialize a runtime dataset config by converting relative `path`
    to absolute path, avoiding remapping to default datasets directory.
    """
    yaml_path = Path(data_yaml)
    data_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    data_root = Path(data_cfg.get("path", ""))
    if not data_root.is_absolute():
        data_root = (yaml_path.parent / data_root).resolve()
    data_cfg["path"] = data_root.as_posix()

    runtime_yaml = ROOT / "outputs" / "_runtime_dataset.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(runtime_yaml)


def _metric_value(metrics) -> float:
    """Extract the primary quality score from Ultralytics validation metrics.

    中文: 从验证结果中提取主要评价指标，优先使用 mAP50-95。
    English: Extract the primary validation score, preferring mAP50-95.
    """
    if metrics is None:
        return float("nan")

    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map"):
        return float(box.map)

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95", "metrics/mAP50"):
            if key in results_dict:
                return float(results_dict[key])

    return float("nan")


def _evaluate_model(model_path: Path, data_yaml: str, args: argparse.Namespace) -> float:
    """Validate one checkpoint and return its score.

    中文: 验证单个权重文件并返回评分。
    English: Validate one checkpoint and return its score.
    """
    model = YOLO(str(model_path))
    metrics = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.25,
        iou=0.6,
        device=args.device,
        workers=args.workers,
        plots=False,
        save=False,
        verbose=False,
    )
    return _metric_value(metrics)


def _promote_best(new_model: Path, old_model: Path, data_yaml: str, args: argparse.Namespace) -> Path:
    """Compare two checkpoints and keep the better one as best.pt.

    中文: 对比新旧模型，保留验证指标更好的权重作为 best.pt。
    English: Compare checkpoints and keep the better one as best.pt.
    """
    old_score = float("nan")
    if old_model.exists():
        old_score = _evaluate_model(old_model, data_yaml, args)

    new_score = _evaluate_model(new_model, data_yaml, args)

    chosen = new_model
    chosen_score = new_score
    if old_model.exists() and not (new_score > old_score or old_score != old_score):
        chosen = old_model
        chosen_score = old_score

    old_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen, old_model)
    print(f"Selected best checkpoint: {chosen}")
    print(f"New model score: {new_score:.4f}" if new_score == new_score else f"New model score: nan")
    if old_model.exists():
        print(f"Old model score: {old_score:.4f}" if old_score == old_score else "Old model score: nan")
    print(f"best.pt updated at: {old_model} (score={chosen_score:.4f})" if chosen_score == chosen_score else f"best.pt updated at: {old_model}")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for training.

    中文: 定义训练命令行参数。
    English: Define command-line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--model", default=str(DEFAULT_PRETRAINED_MODEL))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=15)
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--name", default="train")
    return parser


def main() -> None:
    """Run training with parsed CLI arguments.

    中文: 解析参数并启动训练。
    English: Parse arguments and start model training.
    """
    args = build_parser().parse_args()
    _prepare_runtime()
    model = YOLO(_resolve(args.model))
    runtime_data = _prepare_data_yaml(_resolve(args.data))
    model.train(
        data=runtime_data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        device=args.device,
        workers=args.workers,
        batch=args.batch,
        cache=args.cache,
        amp=args.amp,
        project=_resolve(args.project),
        name=args.name,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", _resolve(args.project)))
    new_best = save_dir / "weights" / "best.pt"
    if new_best.exists():
        _promote_best(new_best, DEFAULT_TRAINED_MODEL, runtime_data, args)
    else:
        print(f"Warning: trained checkpoint not found at {new_best}")


if __name__ == "__main__":
    main()

