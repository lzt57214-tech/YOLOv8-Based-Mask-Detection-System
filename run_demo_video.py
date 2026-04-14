from pathlib import Path

import argparse
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ASSET_VIDEO = ROOT / "assets" / "demo.mp4"
OUTPUT_DIR = ROOT / "outputs"


def get_best_model() -> Path:
    """Prefer newest checkpoint under outputs, fallback to models/trained/best.pt."""
    candidates = [p for p in OUTPUT_DIR.rglob("best.pt") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return ROOT / "models" / "trained" / "best.pt"


def main(view: bool = True) -> None:
    model_path = get_best_model()
    if not model_path.exists():
        raise FileNotFoundError(f"best.pt not found: {model_path}")
    if not ASSET_VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {ASSET_VIDEO}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(ASSET_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {ASSET_VIDEO}")

    print(f"Model: {model_path}")
    if view:
        print("Press 'q' to stop video detection.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=0.25, verbose=False)[0]
        vis = result.plot()
        if view:
            cv2.imshow("Mask Detection - demo.mp4", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if view:
        cv2.destroyAllWindows()
    print("Video detection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mask detection on assets/demo.mp4")
    parser.add_argument("--no-view", action="store_true", help="Run inference without opening a window")
    args = parser.parse_args()
    main(view=not args.no_view)

