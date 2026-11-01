"""Paddle detector only (official tools/eval.py pipeline). Workers receive paths/IDs/hashes, never GT polygons."""
import importlib.metadata
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, canonical, identity, jsonl, read_json, save_json, sha256, write_jsonl

CONFIG = ROOT/'paddle/configs/paddle_v5.json'


def model_identity(model_dir, config_path=CONFIG):
    """Pinned PaddleOCR clone + official config/checkpoint + packages + CS431 code."""
    config = read_json(config_path)
    provenance = read_json(ROOT/'configs/source_provenance.json')
    if sha256(ROOT/'paddle/vendor/eval_det_iou.py') != provenance['paddle_iou_evaluator_sha256']:
        raise RuntimeError('Pinned upstream ICDAR evaluator changed')
    folder = Path(model_dir)
    repo = folder/'PaddleOCR'
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != config['paddleocr_commit']:
        raise RuntimeError(f'PaddleOCR clone must be {config["paddleocr_tag"]} {config["paddleocr_commit"]}, got {head}')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('PaddleOCR clone has modified tracked files')
    if sha256(repo/config['config_relpath']) != config['config_sha256']:
        raise RuntimeError('Official PP-OCRv5_server_det.yml changed')
    if sha256(repo/config['upstream_iou_evaluator_relpath']) != sha256(ROOT/'paddle/vendor/eval_det_iou.py'):
        raise RuntimeError('Vendored evaluator differs from the pinned PaddleOCR evaluator')
    if sha256(folder/'weights'/config['weights_file']) != config['weights_sha256']:
        raise RuntimeError('Official PP-OCRv5_server_det checkpoint changed')
    versions = {name: importlib.metadata.version(name) for name in config['package_versions']}
    if versions != config['package_versions']:
        raise RuntimeError(f'Use the pinned Paddle environment: expected {config["package_versions"]}, got {versions}')
    packages = sorted((d.metadata['Name'].lower().replace('_', '-'), d.version)
                      for d in importlib.metadata.distributions() if d.metadata['Name'])
    code = {name: sha256(ROOT/name) for name in [
        'cs431/common.py', 'cs431/inference.py', 'cs431/ppocr_detector.py', 'cs431/icdar.py',
        'cs431/workflow.py', 'cs431/manifest.py', 'cs431/gate.py',
        'paddle/vendor/eval_det_iou.py']}
    return dict(config_sha256=sha256(config_path), config_content_sha256=identity(config),
                paddleocr_commit=head, official_config_sha256=config['config_sha256'],
                weights_sha256=config['weights_sha256'], package_versions=versions,
                environment_packages_identity=identity(packages), python_version=sys.version,
                code=code)


def parse_output(polygons, scores):
    """Validate detector output: finite quadrilaterals in original-image pixels, scores in [0, 1]."""
    polygons = [[float(v) for v in p] for p in polygons]
    scores = [float(s) for s in scores]
    if len(polygons) != len(scores) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid Paddle output scores')
    if any(len(p) != 8 or any(not math.isfinite(v) for v in p) for p in polygons):
        raise ValueError('Expected finite Paddle quadrilaterals in original-image pixels')
    return polygons, scores


def resume(path):
    path = Path(path)
    if not path.exists():
        return []
    rows, offset = [], 0
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            import json
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1:
                raise ValueError('Corrupt middle of prediction file: ' + str(path))
            path.with_suffix('.interrupted_tail').write_bytes(line)
            with path.open('r+b') as f:
                f.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    else:
        if data and not data.endswith(b'\n'):
            with path.open('ab') as f:
                f.write(b'\n')
    return rows


def worker(tasks_file, out, model_dir, expected_identity, manifest_hash):
    actual_identity = model_identity(model_dir)
    if identity(actual_identity) != expected_identity:
        raise RuntimeError('Worker environment/model differs from Phase 1/controller')
    import paddle
    from .ppocr_detector import OfficialDetector

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
        raise RuntimeError('Worker must see exactly one CUDA GPU; CPU fallback is forbidden')
    config = read_json(CONFIG)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(jsonl(tasks_file))
    fingerprint = dict(identity=expected_identity, manifest_sha256=manifest_hash,
                       tasks_sha256=sha256(tasks_file), gpu=paddle.device.cuda.get_device_name(0))
    meta = out/'identity.json'
    if meta.exists() and read_json(meta) != fingerprint:
        raise RuntimeError('Cannot mix different shards/models/GPUs in a resumed run')
    save_json(meta, fingerprint)
    save_json(out/'runtime.json', dict(
        paddle=paddle.__version__, cuda=paddle.version.cuda(), cudnn=paddle.version.cudnn(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=paddle.device.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True)))
    path = out/'predictions.jsonl'
    old = resume(path)
    expected_ids = {t['sample_id'] for t in tasks}
    seen = {r['sample_id'] for r in old}
    if len(seen) != len(old) or not seen <= expected_ids:
        raise RuntimeError('Duplicate/unexpected resumed predictions')
    if any(r['model_identity_sha256'] != expected_identity or r['manifest_sha256'] != manifest_hash for r in old):
        raise RuntimeError('Resumed prediction identity mismatch')
    todo = [t for t in tasks if t['sample_id'] not in seen]
    if not todo:
        return
    detector = OfficialDetector(Path(model_dir)/'PaddleOCR', config['config_relpath'],
                                Path(model_dir)/'weights'/config['weights_file'], device='gpu:0')
    save_json(out/'detector.json', dict(load=detector.load_report, transforms=detector.transforms,
                                        resize=detector.resize, pipeline=config['inference_pipeline']))
    for task in todo[:1]:
        if sha256(task['path']) != task['image_sha256']:
            raise RuntimeError('Input bytes mismatch before warmup')
        for _ in range(2):
            parse_output(*detector.predict(task['path']))
    with path.open('ab') as f:
        for i, task in enumerate(todo, 1):
            if sha256(task['path']) != task['image_sha256']:
                raise RuntimeError('Input image changed: ' + task['sample_id'])
            paddle.device.cuda.synchronize()
            started = time.perf_counter()
            raw = detector.predict(task['path'])
            paddle.device.cuda.synchronize()
            elapsed = 1000*(time.perf_counter()-started)
            polygons, scores = parse_output(*raw)
            row = {k: v for k, v in task.items() if k not in {'path', 'image_sha256'}}
            row.update(model='paddle', polygons=polygons, scores=scores,
                       manifest_sha256=manifest_hash, model_identity_sha256=expected_identity,
                       checkpoint_sha256=config['weights_sha256'],
                       config_sha256=actual_identity['config_sha256'],
                       inference_ms_with_decode_and_postprocess=elapsed)
            f.write(canonical(row))
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f'PROGRESS {len(old)+i}/{len(tasks)}', flush=True)


def launch(tasks, output, model_dir, model_id, manifest_hash, gpus):
    gpu_ids = [g.strip() for g in gpus.split(',')]
    if not gpu_ids or any(not g.isdigit() for g in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Use unique numeric GPU IDs, e.g. 0,1 or 0')
    if len(gpu_ids) > len(tasks):
        raise ValueError('More GPUs than tasks')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_identity = dict(model_identity=model_id, manifest_sha256=manifest_hash,
                        tasks_sha256=identity(tasks), gpus=gpu_ids)
    existing = out/'inference_identity.json'
    if existing.exists() and read_json(existing) != run_identity:
        raise RuntimeError('Run identity changed; use a new output directory')
    save_json(existing, run_identity)
    processes, handles = [], []
    started = time.perf_counter()
    try:
        for i, gpu in enumerate(gpu_ids):
            task_file = out/f'tasks_gpu{gpu}.jsonl'
            write_jsonl(task_file, tasks[i::len(gpu_ids)])
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            handle = (out/f'gpu{gpu}.log').open('w', encoding='utf-8')
            handles.append(handle)
            cmd = [sys.executable, str(ROOT/'run.py'), '_worker', '--tasks', str(task_file),
                   '--out', str(out/f'gpu{gpu}'), '--model-dir', str(model_dir),
                   '--identity', identity(model_id), '--manifest-hash', manifest_hash]
            processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        last = 0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Paddle worker failed; inspect {out}/gpu*.log. Partial results are not scored.')
            if time.perf_counter()-last >= 30:
                last = time.perf_counter()
                print('INFERENCE running; elapsed seconds:', round(last-started), 'logs:', out, flush=True)
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f'Paddle worker failed; inspect {out}/gpu*.log')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in handles:
            handle.close()
    rows = [r for gpu in gpu_ids for r in jsonl(out/f'gpu{gpu}/predictions.jsonl')]
    ids = {r['sample_id'] for r in rows}
    if len(ids) != len(rows) or ids != {t['sample_id'] for t in tasks}:
        raise RuntimeError('Merged predictions incomplete/extra/duplicate')
    if any(r['model_identity_sha256'] != identity(model_id) or r['manifest_sha256'] != manifest_hash for r in rows):
        raise RuntimeError('Merged predictions contain another model/manifest')
    merged = out/'predictions.jsonl'
    write_jsonl(merged, sorted(rows, key=lambda r: r['sample_id']))
    save_json(out/'timing.json', dict(wall_seconds_this_launch=time.perf_counter()-started,
                                    images=len(rows), gpus=gpu_ids,
                                    note='Worker latency includes decode/preprocess/inference/postprocess; hardware-specific'))
    return merged
