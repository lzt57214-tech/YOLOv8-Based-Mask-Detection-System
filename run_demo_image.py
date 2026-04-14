from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ASSET_IMAGE = ROOT / "assets" / "demo.jpg"
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
    if not ASSET_IMAGE.exists():
        raise FileNotFoundError(f"Image not found: {ASSET_IMAGE}")

    model = YOLO(str(model_path))
    result = model(str(ASSET_IMAGE), conf=0.25, verbose=False)[0]
    vis = result.plot()

    print(f"Model: {model_path}")
    if view:
        cv2.imshow("Mask Detection - demo.jpg", vis)
        print("Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Detection completed in no-view mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mask detection on assets/demo.jpg")
    parser.add_argument("--no-view", action="store_true", help="Run inference without opening a window")
    args = parser.parse_args()
    main(view=not args.no_view)
