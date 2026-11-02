"""Serialized CPU retry of GPU OOM, with unchanged model and preprocessing."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, identity, read_json, save_json, sha256


def gpu_oom(error):
    message = str(error).lower()
    return 'out of memory' in message and ('gpu' in message or 'cuda' in message)


def predict_or_cpu(gpu_predict, cpu_predict, release_gpu):
    reason = None
    try:
        return gpu_predict(), 'gpu', None
    except Exception as error:
        if not gpu_oom(error):
            raise
        reason = str(error)
    # Outside except: release the traceback and its live tensors before retry.
    release_gpu()
    return cpu_predict(reason), 'cpu', reason


@contextmanager
def cpu_slot(path):
    """OS lock releases on process death; do not unlink a shared lock inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def retry_cpu(task, model_dir, expected_identity, out, reason):
    out = Path(out)
    record = out/'cpu_fallback'/f'{identity(task)}.json'
    request = record.with_suffix('.request.json')
    save_json(request, task)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    print('CPU_FALLBACK waiting:', task['sample_id'], flush=True)
    # The child owns the OS lock, so even an interrupted parent cannot allow
    # a second CPU image while its child still holds large tensors in RAM.
    subprocess.run([sys.executable, '-m', 'cs431.cpu_fallback',
                    '--task', str(request), '--model-dir', str(model_dir),
                    '--identity', expected_identity, '--out', str(record),
                    '--lock', str(out.parent/'.cpu_slot.lock')],
                   cwd=ROOT, env=env, check=True)
    result = read_json(record)
    if (result['sample_id'] != task['sample_id'] or
            result['image_sha256'] != task['image_sha256'] or
            result['model_identity_sha256'] != expected_identity):
        raise RuntimeError('CPU result identity mismatch')
    result['gpu_oom'] = reason
    save_json(record, result)
    print('CPU_FALLBACK complete:', task['sample_id'], flush=True)
    return result['polygons'], result['scores']


PRECOMPUTED_ENV = 'CS431_PRECOMPUTED_OOM'


def precomputed_identity():
    """Bind the optional host-precomputed OOM predictions into the model/run identity (Phase 1 receipt + Phase 2)."""
    path = os.environ.get(PRECOMPUTED_ENV)
    if not path:
        return None
    data = read_json(path)
    return dict(file_sha256=sha256(path), image_sha256=sorted(data['entries']),
                detector=data['detector'], environment=data['environment'])


def oom_retry(task, model_dir, expected_identity, out, reason):
    """GPU OOM only. Use the host-precomputed prediction for this exact image (same SHA256) if one was
    provided via CS431_PRECOMPUTED_OOM; otherwise the serialized CPU retry in this environment.
    Returns ((polygons, scores), execution_device)."""
    path = os.environ.get(PRECOMPUTED_ENV)
    if path:
        from .inference import CONFIG
        data, cfg = read_json(path), read_json(CONFIG)
        for key in ('config_sha256', 'weights_sha256', 'paddleocr_commit'):
            if data[key] != cfg[key]:
                raise RuntimeError(f'Precomputed OOM predictions use a different {key}')
        entry = data['entries'].get(task['image_sha256'])
        if entry is not None:
            if entry['sample_id'] != task['sample_id'] or entry['image_sha256'] != task['image_sha256']:
                raise RuntimeError('Precomputed OOM prediction belongs to another sample')
            save_json(Path(out)/'cpu_fallback'/f'{identity(task)}.precomputed.json',
                      dict(entry, gpu_oom=reason, model_identity_sha256=expected_identity,
                           precomputed_file_sha256=sha256(path)))
            print('PRECOMPUTED_OOM used:', task['sample_id'], flush=True)
            return (entry['polygons'], entry['scores']), entry['execution_device']
    return retry_cpu(task, model_dir, expected_identity, out, reason), 'cpu'


def execute(args):
    """Called under an OS lock before importing Paddle or loading weights."""
    from .inference import CONFIG, model_identity, parse_output
    from .ppocr_detector import OfficialDetector
    actual = model_identity(args.model_dir)
    if identity(actual) != args.identity:
        raise RuntimeError('CPU environment/model identity differs from GPU')
    task = read_json(args.task)
    if sha256(task['path']) != task['image_sha256']:
        raise RuntimeError('CPU input image bytes changed')
    print('CPU_FALLBACK running:', task['sample_id'], flush=True)
    cfg = read_json(CONFIG)
    model_dir = Path(args.model_dir)
    detector = OfficialDetector(model_dir/'PaddleOCR', cfg['config_relpath'],
                                model_dir/'weights'/cfg['weights_file'], device='cpu')
    started = time.perf_counter()
    polygons, scores = parse_output(*detector.predict(task['path']))
    save_json(args.out, dict(sample_id=task['sample_id'], image_sha256=task['image_sha256'],
                            model_identity_sha256=args.identity, execution_device='cpu',
                            polygons=polygons, scores=scores, load=detector.load_report,
                            resize=detector.resize, cpu_threads=2,
                            seconds=time.perf_counter()-started))


def main():
    p = argparse.ArgumentParser()
    for field in ('task', 'model-dir', 'identity', 'out', 'lock'):
        p.add_argument('--'+field, required=True)
    args = p.parse_args()
    with cpu_slot(args.lock):
        execute(args)


if __name__ == '__main__':
    main()
