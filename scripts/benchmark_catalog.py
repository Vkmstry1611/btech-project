"""Benchmark new-PDF runs in separate folders; never infer a measured 60-page time."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmarks'))
    parser.add_argument('--max-pages', type=int, default=60)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu:0'], default='auto')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--recognition-batch-size', type=int, default=None)
    parser.add_argument('--baseline-device', choices=['cpu', 'gpu:0'], default='cpu')
    parser.add_argument('--compare', action='store_true', help='Also run sequential preparation with debug output as baseline.')
    args = parser.parse_args()
    if not args.input.is_file(): parser.error('Input PDF does not exist')
    if args.max_pages < 1: parser.error('--max-pages must be positive')
    root = args.output / time.strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'input': str(args.input.resolve()), 'sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'platform': platform.platform(), 'python': sys.version, 'requested_pages': args.max_pages,
                'model': args.model, 'dpi': 200, 'runs': []}
    manifest['versions'] = {}
    for package in ('paddlepaddle', 'paddlepaddle-gpu', 'paddleocr', 'paddlex', 'ollama'):
        try: manifest['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    for mode in (['baseline', 'optimized'] if args.compare else ['optimized']):
        dest = root / mode
        cmd = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('run_final_pipeline.py')),
               '--input', str(args.input), '--output', str(dest), '--backend', 'ollama',
               '--model', args.model, '--device', args.baseline_device if mode == 'baseline' else args.device,
               '--max-pages', str(args.max_pages)]
        if mode != 'baseline' and args.recognition_batch_size is not None:
            cmd += ['--recognition-batch-size', str(args.recognition_batch_size)]
        if mode == 'baseline': cmd += ['--sequential-preprocessing', '--debug-artifacts']
        start = time.perf_counter()
        with (root / f'{mode}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        record = {'mode': mode, 'wall_including_startup_s': time.perf_counter()-start, 'exit_code': run.returncode}
        metric = dest / 'metrics' / args.input.stem / 'pipeline.json'
        if metric.exists(): record['pipeline'] = json.loads(metric.read_text(encoding='utf-8'))
        manifest['runs'].append(record)
        (root / 'benchmark.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2), flush=True)
        if run.returncode: raise SystemExit(run.returncode)
    print('Benchmark evidence:', root)


if __name__ == '__main__': main()
