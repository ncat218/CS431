"""Run from the repo root. No GPU framework imports for help/status/shared checks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from cs431.common import ROOT, run_lock


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'monkey':
        return subprocess.run([sys.executable, str(ROOT/'monkey/run.py'), *argv[1:]],
                              cwd=ROOT/'monkey').returncode
    p = argparse.ArgumentParser(description='CS431: detector reproduction and paired robustness evaluation')
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('monkey', help='Independent Monkey workspace: status, shared, phase2, phase3')
    shared = commands.add_parser('shared', help='Verify or unpack the frozen shared manifest')
    shared.add_argument('action', choices=['verify', 'unpack'])
    paddle = commands.add_parser('paddle')
    actions = paddle.add_subparsers(dest='action', required=True)
    actions.add_parser('status')
    for name in ('phase1', 'phase2'):
        stage = actions.add_parser(name)
        stage.add_argument('--model-dir', default=str(ROOT/'paddle/models/PP-OCRv5_server_det'))
        stage.add_argument('--gpus', default='0,1', help='Kaggle: 0,1; Colab A100: 0')
        stage.add_argument('--out', default=str(ROOT/f'paddle/{name}/report'))
        if name == 'phase1':
            stage.add_argument('--images', required=True)
            stage.add_argument('--gt', required=True)
            stage.add_argument('--dataset-source', required=True, help='Actual source URL/version/provenance for attached ICDAR2015 test')
        else:
            stage.add_argument('--locations', required=True, help='Local JSON containing clean_root and degraded_root')
    review = actions.add_parser('review')
    review.add_argument('--run', default=str(ROOT/'paddle/phase1/report'))
    review.add_argument('--reviewer', required=True)
    review.add_argument('--notes', required=True, help='Your assessment of metrics, overlays and limitations')
    compare = commands.add_parser('compare')
    compare.add_argument('--paddle-metrics', required=True)
    compare.add_argument('--monkey-metrics', default=str(ROOT/'monkey/phase2/report/metrics.json'))
    compare.add_argument('--out', default=str(ROOT/'comparison/phase2.md'))
    worker = commands.add_parser('_worker', help=argparse.SUPPRESS)
    for name in ('tasks', 'out', 'model-dir', 'identity', 'manifest-hash'):
        worker.add_argument('--'+name, required=True)
    args = p.parse_args(argv)
    if args.command == 'shared':
        from cs431.manifest import unpack, verify_shared
        print(unpack() if args.action == 'unpack' else json.dumps(verify_shared(), indent=2))
    elif args.command == 'paddle':
        from cs431 import gate
        if args.action == 'status':
            try:
                receipt = gate.require_phase1()
                print('REVIEWED: '+receipt['reviewer']+'. Model/environment identity is rechecked at Phase 2 launch.')
            except (RuntimeError, FileNotFoundError, KeyError, ValueError) as e:
                print('BLOCKED:', e)
                return 2
        elif args.action == 'review':
            print(json.dumps(gate.approve(args.run, args.reviewer, args.notes), indent=2))
        else:
            # Gate before importing modules that need numpy/Pillow/Shapely.
            if args.action == 'phase2':
                gate.require_phase1()
            from cs431 import workflow
            with run_lock(args.out):
                getattr(workflow, args.action)(args)
    elif args.command == 'compare':
        from cs431.workflow import compare as compare_results
        print(compare_results(args.paddle_metrics, args.monkey_metrics, args.out))
    else:
        from cs431.inference import worker as run_worker
        run_worker(args.tasks, args.out, args.model_dir, args.identity, args.manifest_hash)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
