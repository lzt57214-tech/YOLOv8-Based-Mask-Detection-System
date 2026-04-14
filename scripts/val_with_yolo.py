import argparse
import json
from pathlib import Path
import shutil

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import yaml


def find_latest_metrics(run_root: Path):
    candidates = list(run_root.glob('runs/val/*/metrics.json'))
    if not candidates:
        candidates = list(run_root.glob('runs/detect/val*/metrics.json'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_metrics_from_result(res):
    """Serialize common Ultralytics val metrics into a plain dict."""
    out = {}
    try:
        rd = getattr(res, 'results_dict', None)
        if isinstance(rd, dict):
            out['results_dict'] = rd
    except Exception:
        pass

    try:
        box = getattr(res, 'box', None)
        if box is not None:
            out['box'] = {
                'map50': float(getattr(box, 'map50', 0.0)),
                'map': float(getattr(box, 'map', 0.0)),
                'mp': float(getattr(box, 'mp', 0.0)),
                'mr': float(getattr(box, 'mr', 0.0)),
            }
            if hasattr(box, 'maps'):
                try:
                    out['box']['maps'] = [float(x) for x in list(box.maps)]
                except Exception:
                    pass
    except Exception:
        pass
    return out


def run_val_with_cli(model: str, data: str, imgsz: int, batch: int):
    # fallback to CLI if ultralytics package not importable
    import subprocess
    cmd = ["yolo", "val", f"model={model}", f"data={data}", f"imgsz={imgsz}", f"batch={batch}"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _expand_images_from_yaml(data_yaml: Path):
    """Return absolute image file list from dataset.yaml val field."""
    with open(data_yaml, 'r', encoding='utf-8') as fh:
        d = yaml.safe_load(fh)

    dataset_root = d.get('path')
    if dataset_root:
        dataset_root = (data_yaml.parent / str(dataset_root)).resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    val_field = d.get('val')
    val_entries = val_field if isinstance(val_field, list) else [val_field]
    val_entries = [v for v in val_entries if v]

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    all_images = []
    for entry in val_entries:
        p = Path(str(entry))
        if not p.is_absolute():
            p = (dataset_root / p).resolve()
        if p.is_dir():
            for ext in exts:
                all_images.extend([str(x.resolve()) for x in p.rglob(ext)])
        elif p.is_file():
            all_images.append(str(p.resolve()))
    # de-duplicate and stable order
    return sorted(set(all_images))


def main():
    parser = argparse.ArgumentParser(description='Run YOLO validation and save metrics and predictions')
    parser.add_argument('--model', required=True, help='path to model weights')
    parser.add_argument('--data', required=True, help='dataset yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--tag', type=str, default='run')
    parser.add_argument('--out-dir', type=str, default='outputs/val')
    parser.add_argument('--device', type=str, default=None, help='device id or cpu')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_src = None
    val_metrics = None
    try:
        if YOLO is None:
            print('ultralytics package not available, falling back to CLI')
            run_val_with_cli(args.model, args.data, args.imgsz, args.batch)
        else:
            model = YOLO(args.model)
            # run val
            print('Running model.val ...')
            kwargs = {}
            if args.device:
                kwargs['device'] = args.device
            res = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, **kwargs)
            val_metrics = _build_metrics_from_result(res)
    except Exception as e:
        print('Validation run failed:', e)

    # try to find latest metrics.json from runs/val
    root = Path.cwd()
    metrics_src = find_latest_metrics(root)
    dst = out_dir / f"{args.tag}_results.json"
    if metrics_src is not None:
        dst = out_dir / f"{args.tag}_results.json"
        shutil.copy(metrics_src, dst)
        print(f'Copied metrics to {dst}')
    elif val_metrics is not None:
        with open(dst, 'w', encoding='utf-8') as fh:
            json.dump(val_metrics, fh, ensure_ascii=False, indent=2)
        print(f'Wrote serialized metrics to {dst}')
    else:
        print('Could not find metrics output. Please check yolo installation or run manually.')

    # Additionally, generate per-image predictions by running model.predict on val images (if ultralytics available)
    preds_out = out_dir / f"{args.tag}_preds.json"
    try:
        if YOLO is None:
            print('Skipping per-image prediction (ultralytics library unavailable).')
            return
        # parse dataset.yaml to find val images (resolved absolute paths)
        val_images = _expand_images_from_yaml(Path(args.data))

        model = YOLO(args.model)
        all_preds = []
        print(f'Predicting on {len(val_images)} val images (this may take a while)...')
        for img in val_images:
            try:
                r = model.predict(source=img, imgsz=args.imgsz, device=args.device or None, verbose=False)
            except Exception as e:
                print('predict failed for', img, e)
                continue
            # r is a list of Results; take first
            if not r:
                all_preds.append({'image': str(Path(img).resolve()), 'preds': []})
                continue
            res = r[0]
            preds = []
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                # older versions may use res.boxes
                all_preds.append({'image': str(Path(img).resolve()), 'preds': []})
                continue
            for b in boxes:
                # boxes.xyxyn or boxes.xyxy may be tensors or numpy
                try:
                    xyxy = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    cls = int(b.cls[0])
                except Exception:
                    # fallback for different API
                    try:
                        xyxy = b.xyxy.tolist()
                        conf = float(b.conf)
                        cls = int(b.cls)
                    except Exception:
                        continue
                preds.append({'class': cls, 'conf': conf, 'xyxy': xyxy})
            all_preds.append({'image': str(Path(img).resolve()), 'preds': preds})

        with open(preds_out, 'w', encoding='utf-8') as fh:
            json.dump(all_preds, fh, ensure_ascii=False, indent=2)
        print(f'Wrote predictions to {preds_out}')
    except Exception as e:
        print('Failed to generate per-image predictions:', e)


if __name__ == '__main__':
    main()
