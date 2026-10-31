"""Local review receipt: fresh clones cannot inherit someone else's Phase 1."""
import math
import uuid
from pathlib import Path

from .common import ROOT, identity, jsonl, now, read_json, save_json, sha256


def clone_id(root=ROOT):
    path = Path(root) / '.local/clone.json'
    if not path.exists():
        save_json(path, dict(id=str(uuid.uuid4()), created_utc=now()))
    return read_json(path)['id']


def verify_result(run, root=ROOT):
    run = Path(run).resolve()
    report = read_json(run / 'phase1_result.json')
    if report['clone_id'] != clone_id(root):
        raise RuntimeError('Phase 1 belongs to another clone; run ICDAR2015 here first')
    if report['status'] != 'EVALUATED_REQUIRES_REVIEW':
        raise RuntimeError('Phase 1 evaluation is incomplete')
    if report['benchmark'] != 'ICDAR2015 test' or report['images'] != 500:
        raise RuntimeError('Gate requires the full 500-image ICDAR2015 test split')
    required = {'metrics.json', 'dataset_inventory.json', 'config_snapshot.json', 'model_identity.json',
                'per_image.jsonl', 'inference/predictions.jsonl', 'pip_freeze.txt', 'REVIEW_ME.md'}
    if not required <= set(report['artifacts']):
        raise RuntimeError('Missing required Phase 1 evidence')
    if len([p for p in report['artifacts'] if p.startswith('overlays/') and p.endswith('.jpg')]) < 20:
        raise RuntimeError('Phase 1 needs at least 20 review overlays')
    for relpath, expected in report['artifacts'].items():
        path = (run / relpath).resolve()
        if not path.is_relative_to(run) or sha256(path) != expected:
            raise RuntimeError('Phase 1 artifact changed: ' + relpath)
    metrics = read_json(run / 'metrics.json')
    if metrics['images'] != 500 or metrics['exact_paddle_83_8_reproduction'] != 'NOT VERIFIED':
        raise RuntimeError('Wrong Phase 1 scope')
    for name in ('precision', 'recall', 'hmean'):
        v = metrics['measured'][name]
        if not math.isfinite(v) or not 0 < v <= 1:
            raise RuntimeError('Degenerate/non-finite Phase 1 metrics; investigate before review')
    counts = metrics['counts']
    if not 0 < counts['matched'] <= min(counts['gt'], counts['predictions']):
        raise RuntimeError('Invalid Phase 1 counts')
    p, r = counts['matched']/counts['predictions'], counts['matched']/counts['gt']
    for name, expected in dict(precision=p, recall=r, hmean=2*p*r/(p+r)).items():
        if abs(metrics['measured'][name] - expected) > 1e-12:
            raise RuntimeError('Phase 1 metrics do not match raw counts')
    expected_ids = {f'img_{i}' for i in range(1, 501)}
    inventory = read_json(run/'dataset_inventory.json')
    records = inventory['records']
    if len(records) != 500 or {r['image_id'] for r in records} != expected_ids:
        raise RuntimeError('Phase 1 inventory must contain all 500 test IDs')
    if inventory['dataset_identity'] != identity(records):
        raise RuntimeError('Dataset inventory identity mismatch')
    rows = list(jsonl(run/'inference/predictions.jsonl'))
    if len(rows) != 500 or {r['sample_id'] for r in rows} != expected_ids:
        raise RuntimeError('Phase 1 raw predictions must cover every test image once')
    if any(r['model'] != 'paddle' or r['model_identity_sha256'] != identity(report['model_identity']) for r in rows):
        raise RuntimeError('Phase 1 predictions use another model')
    run_hash = identity(dict(dataset=inventory, clone_id=report['clone_id']))
    if any(r['manifest_sha256'] != run_hash for r in rows):
        raise RuntimeError('Phase 1 predictions belong to another clone/dataset')
    per_image = list(jsonl(run/'per_image.jsonl'))
    if len(per_image) != 500 or {r['sample_id'] for r in per_image} != expected_ids:
        raise RuntimeError('Phase 1 per-image metrics missing or duplicated')
    for field in ('gt', 'predictions', 'matched'):
        if sum(r[field] for r in per_image) != counts[field]:
            raise RuntimeError('Per-image counts differ from Phase 1 metrics')
    if read_json(run/'model_identity.json') != report['model_identity']:
        raise RuntimeError('Model identity evidence mismatch')
    if identity(read_json(run/'config_snapshot.json')) != report['model_identity']['config_content_sha256']:
        raise RuntimeError('Configuration evidence mismatch')
    return report


def approve(run, reviewer, notes, root=ROOT):
    if not reviewer.strip() or len(notes.strip()) < 20:
        raise ValueError('Enter reviewer and >=20 characters describing metrics, overlays and any limitations')
    report = verify_result(run, root)
    receipt = dict(clone_id=clone_id(root), reviewed_utc=now(), reviewer=reviewer.strip(),
                   notes=notes.strip(), conclusion='ACCEPTED_EXTERNAL_ICDAR2015_VALIDATION',
                   exact_paddle_83_8_reproduction='NOT VERIFIED',
                   run=str(Path(run).resolve()), result_sha256=sha256(Path(run)/'phase1_result.json'),
                   model_identity=report['model_identity'])
    save_json(Path(root)/'.local/paddle_phase1_review.json', receipt)
    return receipt


def require_phase1(model_identity=None, root=ROOT):
    path = Path(root)/'.local/paddle_phase1_review.json'
    if not path.exists():
        raise RuntimeError('PHASE 2 BLOCKED: run Paddle Phase 1 on ICDAR2015, inspect results, then paddle review')
    receipt = read_json(path)
    if receipt['clone_id'] != clone_id(root):
        raise RuntimeError('Review receipt belongs to another clone')
    report = verify_result(receipt['run'], root)
    if sha256(Path(receipt['run'])/'phase1_result.json') != receipt['result_sha256']:
        raise RuntimeError('Reviewed Phase 1 report changed')
    if receipt['conclusion'] != 'ACCEPTED_EXTERNAL_ICDAR2015_VALIDATION':
        raise RuntimeError('Phase 1 not accepted')
    if identity(report['model_identity']) != identity(receipt['model_identity']):
        raise RuntimeError('Receipt/model identity mismatch')
    if model_identity is not None and identity(model_identity) != identity(receipt['model_identity']):
        raise RuntimeError('Model/config/code/environment changed since Phase 1; rerun validation and review')
    return receipt
