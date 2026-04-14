from __future__ import annotations
"""Unified CLI for yolo_mask_detector.

中文: yolo_mask_detector 的统一命令行入口，包含训练、验证、推理和 GUI。
English: Unified CLI entry for training, validation, inference, and GUI.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO, settings as ul_settings

from run_gui import main as run_gui_main

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset.yaml"
DEFAULT_OUTPUT = ROOT / "outputs"
DEFAULT_PRETRAINED_MODEL = ROOT / "models" / "pretrained" / "yolov8n.pt"
DEFAULT_IMAGE_SOURCE = ROOT / "assets" / "demo.jpg"


def _default_trained_model() -> Path:
    """Choose the newest trained checkpoint.

    中文: 优先使用 outputs 下最新生成的 best.pt，避免命令行和 GUI 误用旧权重。
    English: Prefer the newest best.pt under outputs to avoid stale weights.
    """
    candidates = [p for p in (ROOT / "outputs").rglob("best.pt") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return ROOT / "models" / "trained" / "best.pt"


def _resolve(path: str | Path) -> str:
    """Resolve relative paths against project root.

    中文: 将相对路径转换为以 yolo_mask_detector 为基准的绝对路径。
    English: Convert relative paths to absolute paths based on yolo_mask_detector.
    """
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return str(p)


def _prepare_runtime() -> None:
    """Prepare Ultralytics runtime defaults.

    中文: 固定 datasets_dir 为项目根目录，保证 dataset.yaml 的相对路径可移植。
    English: Pin datasets_dir to the project root for portable relative dataset paths.
    """
    ul_settings.update({"datasets_dir": str(ROOT)})


def cmd_train(args: argparse.Namespace) -> None:
    """Train model.

    中文: 执行训练。
    English: Execute training.
    """
    _prepare_runtime()
    model = YOLO(_resolve(args.model))
    model.train(
        data=_resolve(args.data),
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


def cmd_val(args: argparse.Namespace) -> None:
    """Validate model.

    中文: 执行验证。
    English: Execute validation.
    """
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


def cmd_predict(args: argparse.Namespace) -> None:
    """Run image prediction.

    中文: 执行图片推理。
    English: Run image inference.
    """
    model = YOLO(_resolve(args.model))
    results = model.predict(
        source=_resolve(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=_resolve(args.project),
        name=args.name,
        save=True,
    )
    if results:
        print(f"Saved results to: {_resolve(args.project)}")


def cmd_video(args: argparse.Namespace) -> None:
    """Run video or webcam prediction.

    中文: 执行视频/摄像头实时推理。
    English: Run real-time inference for video/webcam.
    """
    model = YOLO(_resolve(args.model))
    source = 0 if args.source == "0" else _resolve(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=args.conf, iou=args.iou)[0]
        vis = result.plot()
        cv2.imshow("yolo-mask-detector", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_gui(_: argparse.Namespace) -> None:
    """Launch GUI.

    中文: 打开图形化界面。
    English: Open graphical interface.
    """
    run_gui_main()


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser.

    中文: 构建命令行参数解析器。
    English: Build command-line parser.
    """
    parser = argparse.ArgumentParser(description="Minimal YOLOv8 project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a detection model")
    train.add_argument("--model", default=str(DEFAULT_PRETRAINED_MODEL))
    train.add_argument("--data", default=str(DEFAULT_DATA))
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=768)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--cache", action="store_true")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=15)
    train.add_argument("--project", default=str(DEFAULT_OUTPUT))
    train.add_argument("--name", default="train")
    train.set_defaults(func=cmd_train)

    val = sub.add_parser("val", help="Validate a trained model")
    val.add_argument("--model", default=str(_default_trained_model()))
    val.add_argument("--data", default=str(DEFAULT_DATA))
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=4)
    val.add_argument("--workers", type=int, default=0)
    val.add_argument("--device", default="cpu")
    val.add_argument("--conf", type=float, default=0.25)
    val.add_argument("--iou", type=float, default=0.6)
    val.add_argument("--project", default=str(DEFAULT_OUTPUT))
    val.add_argument("--name", default="val")
    val.set_defaults(func=cmd_val)

    predict = sub.add_parser("predict", help="Run image inference")
    predict.add_argument("--model", default=str(_default_trained_model()))
    predict.add_argument("--source", default=str(DEFAULT_IMAGE_SOURCE))
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--project", default=str(DEFAULT_OUTPUT))
    predict.add_argument("--name", default="predict")
    predict.set_defaults(func=cmd_predict)

    video = sub.add_parser("video", help="Run webcam or video inference")
    video.add_argument("--model", default=str(_default_trained_model()))
    video.add_argument("--source", default="0", help="0 for webcam, or video file path")
    video.add_argument("--conf", type=float, default=0.25)
    video.add_argument("--iou", type=float, default=0.7)
    video.set_defaults(func=cmd_video)

    gui = sub.add_parser("gui", help="Open the graphical interface")
    gui.set_defaults(func=cmd_gui)

    return parser


def main() -> None:
    """CLI program entry.

    中文: 命令行程序入口。
    English: Program entry for CLI mode.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

