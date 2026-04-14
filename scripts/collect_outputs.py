import json
from pathlib import Path
import argparse


def load_json_if_exists(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/collected_results.json')
    parser.add_argument('--outputs-dir', default='outputs')
    args = parser.parse_args()

    root = Path(args.outputs_dir)
    collected = {
        'project': Path.cwd().name,
        'labels_count': None,
        'val_results': {},
        'predictions': {},
        'confusion_matrices': {},
        'benchmarks': {},
        'notes': ''
    }

    # labels count
    lc = load_json_if_exists(root / 'labels_count.json') or load_json_if_exists(root / 'labels_count' / 'labels_count.json')
    if lc:
        collected['labels_count'] = lc

    # val results
    val_dir = root / 'val'
    if val_dir.exists():
        for p in val_dir.glob('*_results.json'):
            tag = p.stem.replace('_results', '')
            collected['val_results'][tag] = load_json_if_exists(p) or {'file': str(p)}
        for p in val_dir.glob('*_preds.json'):
            tag = p.stem.replace('_preds', '')
            collected['predictions'][tag] = str(p)
        for p in val_dir.glob('*_confusion.json'):
            tag = p.stem.replace('_confusion', '')
            collected['confusion_matrices'].setdefault(tag, {})['json'] = load_json_if_exists(p)
        for p in val_dir.glob('*_confusion.csv'):
            tag = p.stem.replace('_confusion', '')
            collected['confusion_matrices'].setdefault(tag, {})['csv'] = str(p)

    # benchmarks
    bench_dir = root / 'benchmark'
    if bench_dir.exists():
        for p in bench_dir.glob('*.json'):
            collected['benchmarks'][p.stem] = load_json_if_exists(p) or str(p)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Wrote aggregated results to', out_path)


if __name__ == '__main__':
    main()
