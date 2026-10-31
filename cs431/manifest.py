"""Unpack the reviewed artifact; never rebuild a model-specific intersection."""
import gzip
import shutil
from pathlib import Path

from .common import ROOT, jsonl, read_json, sha256


def verify_shared(manifest=None, root=ROOT):
    root = Path(root)
    lock = read_json(root / 'configs/shared_lock.json')
    checks = {
        'compressed_manifest_sha256': root / 'shared/frozen/shared_manifest_v1.jsonl.gz',
        'protocol_sha256': root / 'shared/frozen/evaluation_protocol.json',
        'evaluator_sha256': root / 'shared/evaluate_shared.py',
        'builder_sha256': root / 'shared/shared_manifest.py',
        'public_notebook_sha256': root / 'shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb',
    }
    if manifest is not None:
        checks['manifest_sha256'] = Path(manifest)
    for key, path in checks.items():
        if sha256(path) != lock[key]:
            raise RuntimeError(f'Frozen shared artifact changed: {path} ({key})')
    return lock


def unpack(destination=None, root=ROOT):
    root = Path(root)
    lock = verify_shared(root=root)
    archive = root / 'shared/frozen/shared_manifest_v1.jsonl.gz'
    if sha256(archive) != lock['compressed_manifest_sha256']:
        raise RuntimeError('Compressed manifest does not match reviewed artifact')
    path = Path(destination) if destination else root / 'outputs/shared/shared_manifest_v1.jsonl'
    if path.exists():
        verify_shared(path, root)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.jsonl.unpacking')
    try:
        with gzip.open(archive, 'rb') as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        verify_shared(temp, root)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def build_tasks(manifest, locations):
    clean, degraded, variants = {}, [], set()
    for row in jsonl(manifest):
        source = row['source_image_id']
        task = dict(sample_id='clean::' + source, source_image_id=source,
                    condition='clean', path=str(Path(locations['clean_root']) / row['clean_relpath']),
                    image_sha256=row['clean_sha256'])
        if source in clean and clean[source] != task:
            raise ValueError('Inconsistent clean image across variants')
        clean[source] = task
        variant = row['variant_id']
        if variant in variants:
            raise ValueError('Duplicate variant ID')
        variants.add(variant)
        degraded.append(dict(sample_id='deg::' + variant, source_image_id=source,
                             variant_id=variant, condition='degraded',
                             path=str(Path(locations['degraded_root']) / row['degraded_relpath']),
                             image_sha256=row['degraded_sha256']))
    return [clean[k] for k in sorted(clean)] + degraded
