import argparse
import time
import statistics
import json
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--images', required=True, help='single image path or directory')
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', default=None, help='cpu or cuda:0 or 0')
    parser.add_argument('--out', default='outputs/benchmark/benchmark.json')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if YOLO is None:
        raise RuntimeError('ultralytics is required for benchmark_inference.py')

    model = YOLO(args.model)

    imgs = []
    p = Path(args.images)
    if p.is_dir():
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            imgs += list(p.rglob(ext))
    elif p.is_file():
        imgs = [p]
    else:
        raise FileNotFoundError('images path not found: ' + args.images)

    if not imgs:
        raise RuntimeError('no images found for benchmarking')

    # choose first image for repeated runs
    img = str(imgs[0])

    # device handling
    device = args.device
    if device is None:
        device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'

    print(f'Benchmarking on device={device}, image={img}, runs={args.runs}, warmup={args.warmup}')

    # warmup
    for _ in range(args.warmup):
        _ = model.predict(source=img, device=device)

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        _ = model.predict(source=img, device=device)
        if torch is not None and device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg_ms = statistics.mean(times)
    p50 = statistics.median(times)
    p90 = sorted(times)[int(len(times) * 0.9) - 1]
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    out = {
        'device': device,
        'image': img,
        'runs': args.runs,
        'warmup': args.warmup,
        'avg_latency_ms': avg_ms,
        'p50_ms': p50,
        'p90_ms': p90,
        'fps': fps,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('Wrote benchmark to', out_path)


if __name__ == '__main__':
    main()
