"""Independent Monkey entrypoint; runtime is isolated, shared benchmark bytes are not."""
import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE.parent
sys.path.insert(0, str(ROOT))

from cs431.common import jsonl, read_json, sha256
from cs431.manifest import unpack, verify_shared


def local_settings():
    path = WORKSPACE/'workspace.local.json'
    return read_json(path) if path.exists() else {}


def safe_runtime(path):
    path = Path(path).expanduser().resolve()
    # All writes belong to Monkey runtime, never benchmark/source/Paddle state.
    allowed_local = WORKSPACE/'.runtime'
    if path == allowed_local or path.is_relative_to(allowed_local):
        return path
    if path == ROOT or path.is_relative_to(ROOT) or ROOT.is_relative_to(path):
        raise ValueError('Inside CS431, Monkey runtime must be under monkey/.runtime')
    if any(part == '_workspace_monkeyocr_text_detection_phase3' for part in path.parts):
        raise ValueError('Use a separate CS431 runtime, not the legacy H200 repository workspace')
    return path


def runtime_root(explicit='', settings=None):
    settings = settings or {}
    value = explicit or os.environ.get('CS431_MONKEY_WORKSPACE', '') or settings.get('runtime_root', '')
    if not value:
        datastore = Path('/datastore')/getpass.getuser()
        value = datastore/'_workspace_cs431_monkey' if datastore.is_dir() else WORKSPACE/'.runtime'
    return safe_runtime(value)


def phase3_command(args, runtime, settings):
    command = [sys.executable, str(WORKSPACE/'phase3/run_all.py'),
               '--workspace', str(runtime/'phase3')]
    if args.config:
        command += ['--config', str(Path(args.config).resolve())]
    parent = args.dataset_parent or settings.get('dataset_parent', '')
    if parent:
        command += ['--dataset-parent', parent]
    if args.plan_only:
        command += ['--plan-only']
    return command


def phase2_command(args, runtime, python):
    lock = verify_shared()
    frozen = unpack(runtime/'shared/shared_manifest_v1.jsonl')
    pins = read_json(WORKSPACE/'phase3/configs/verified_phase1_phase2_sources.json')
    for old, key in [('phase2_manifest_sha256','manifest_sha256'),
                     ('evaluation_protocol_sha256','protocol_sha256'),
                     ('evaluator_sha256','evaluator_sha256')]:
        if pins[old] != lock[key]:
            raise RuntimeError('Monkey and shared benchmark locks disagree: '+key)
    if args.action == 'infer':
        for name, key in [('checkpoint','checkpoint_sha256'), ('config','official_config_sha256')]:
            if sha256(getattr(args, name)) != pins[key]:
                raise RuntimeError('Use the verified Phase 1 '+name+'; identity mismatch')
        locations = Path(args.locations).resolve()
        roots = read_json(locations)
        if not all(roots.get(key) for key in ('clean_root','degraded_root')):
            raise ValueError('Locations JSON needs clean_root and degraded_root')
        return [python, str(WORKSPACE/'phase2/run_monkey.py'),
                '--manifest', str(frozen), '--locations', str(locations), '--gpus', args.gpus,
                '--output-dir', str(runtime/'phase2/inference'),
                '--config', str(Path(args.config).resolve()),
                '--checkpoint', str(Path(args.checkpoint).resolve()),
                '--config-hash', pins['official_config_sha256'],
                '--checkpoint-hash', pins['checkpoint_sha256']]
    predictions = Path(args.predictions).resolve() if args.predictions else runtime/'phase2/inference/monkey_predictions.jsonl'
    for row in jsonl(predictions):
        if (row.get('model') != 'monkey' or row.get('manifest_sha256') != lock['manifest_sha256'] or
                row.get('checkpoint_sha256') != pins['checkpoint_sha256'] or
                row.get('config_sha256') != pins['official_config_sha256']):
            raise RuntimeError('Phase 2 requires original Monkey predictions on the frozen full master')
    return [python, str(ROOT/'shared/evaluate_shared.py'), '--manifest', str(frozen),
            '--protocol', str(ROOT/'shared/frozen/evaluation_protocol.json'),
            '--predictions', str(predictions), '--output-dir', str(runtime/'phase2/metrics'),
            '--bootstrap', '1000']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', default='', help='Separate Monkey runtime directory')
    parser.add_argument('--python', default='', help='Verified Monkey Python interpreter for Phase 2')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status')
    shared = commands.add_parser('shared')
    shared.add_argument('action', choices=['verify','unpack'])
    phase3 = commands.add_parser('phase3')
    phase3.add_argument('--plan-only', action='store_true')
    phase3.add_argument('--dataset-parent', default='')
    phase3.add_argument('--config', default='')
    phase2 = commands.add_parser('phase2')
    actions = phase2.add_subparsers(dest='action', required=True)
    infer = actions.add_parser('infer')
    infer.add_argument('--config', required=True)
    infer.add_argument('--checkpoint', required=True)
    infer.add_argument('--locations', required=True)
    infer.add_argument('--gpus', default='0')
    evaluate = actions.add_parser('evaluate')
    evaluate.add_argument('--predictions', default='')
    args = parser.parse_args(argv)
    settings = local_settings()
    runtime = runtime_root(args.runtime_root, settings)
    if args.command == 'status':
        print(json.dumps(dict(workspace=str(WORKSPACE), runtime_root=str(runtime),
                              phase1_notebook=str(WORKSPACE/'phase1/MonkeyOCRv2_Phase1_ICDAR2015.ipynb'),
                              reports={f'phase{i}':str(WORKSPACE/f'phase{i}/report') for i in (1,2,3)},
                              shared_root=str(ROOT/'shared'),
                              phase2_reference_sha256=read_json(ROOT/'configs/shared_lock.json')['manifest_sha256'],
                              phase3_runner=str(WORKSPACE/'phase3/run_all.py'),
                              environment='Monkey only; Phase 3 bootstraps its own runtime; Phase 2 uses --python'), indent=2))
        return 0
    if args.command == 'shared':
        print(json.dumps(verify_shared(), indent=2) if args.action == 'verify' else
              unpack(runtime/'shared/shared_manifest_v1.jsonl'))
        return 0
    if args.command == 'phase3':
        command = phase3_command(args, runtime, settings)
    else:
        python = args.python or settings.get('python', '') or sys.executable
        command = phase2_command(args, runtime, python)
    return subprocess.run(command, cwd=WORKSPACE).returncode


if __name__ == '__main__':
    raise SystemExit(main())
