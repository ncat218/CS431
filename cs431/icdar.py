"""Explicit ICDAR2015 test IDs and unmodified PaddleOCR IoU evaluator (identical in v3.0.3 and v3.3.3)."""
import math
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon

from .common import ROOT, identity, jsonl, module, sha256

EVALUATOR = ROOT / 'paddle/vendor/eval_det_iou.py'


def dataset(images, gt):
    images, gt = Path(images).resolve(), Path(gt).resolve()
    image_files = {}
    for p in images.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if p.stem in image_files:
                raise ValueError(f'Duplicate image name: {p.stem}')
            image_files[p.stem] = p
    expected = {f'img_{i}' for i in range(1, 501)}
    if set(image_files) != expected:
        raise ValueError('Phase 1 requires exactly ICDAR2015 TEST img_1 .. img_500; no partial/train split')
    gt_files = {}
    for p in gt.rglob('gt_img_*.txt'):
        key = p.stem.removeprefix('gt_')
        if key in gt_files:
            raise ValueError('Duplicate GT file: ' + key)
        gt_files[key] = p
    if set(gt_files) != expected:
        raise ValueError('Need all 500 matching gt_img_N.txt files, with no extra IDs')
    records, tasks = [], []
    for key in sorted(expected):
        image, annotation = image_files[key], gt_files[key]
        with Image.open(image) as im:
            if im.getexif().get(274, 1) not in (None, 1):
                raise ValueError('Unexpected EXIF orientation: ' + key)
            im.convert('RGB').load()
        entries = parse_gt(annotation)
        if not entries:
            raise ValueError('Empty ICDAR GT: ' + key)
        records.append(dict(image_id=key, image_sha256=sha256(image), gt_sha256=sha256(annotation)))
        tasks.append(dict(sample_id=key, path=str(image), image_sha256=sha256(image), condition='icdar2015_test'))
    return tasks, dict(benchmark='ICDAR2015 test', images=500, records=records,
                       dataset_identity=identity(records)), gt_files


def parse_gt(path):
    entries = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        parts = line.split(',', 8)
        if len(parts) != 9:
            raise ValueError(f'Invalid ICDAR annotation: {path}')
        points = [float(v) for v in parts[:8]]
        polygon = list(zip(points[::2], points[1::2]))
        geom = Polygon(polygon)
        if not all(math.isfinite(v) for v in points) or not geom.is_valid or geom.area <= 0:
            raise ValueError(f'Invalid GT polygon; audit instead of silently dropping: {path}')
        entries.append(dict(points=polygon, ignore=parts[8].strip() == '###'))
    return entries


def evaluate(predictions, gt_files):
    evaluator = module(EVALUATOR, 'paddle_official_iou').DetectionIoUEvaluator()
    rows = list(jsonl(predictions))
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(gt_files):
        raise ValueError('ICDAR predictions incomplete, extra or duplicate')
    results, per_image = [], []
    for key in sorted(gt_files):
        row = by_id[key]
        if len(row['polygons']) != len(row['scores']):
            raise ValueError('Mismatched polygons/scores')
        pred = []
        for coords, score in zip(row['polygons'], row['scores']):
            if len(coords) != 8 or not all(math.isfinite(v) for v in coords):
                raise ValueError('ICDAR expects a finite quadrilateral')
            points = list(zip(coords[::2], coords[1::2]))
            geom = Polygon(points)
            if not geom.is_valid or geom.area <= 0 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detector output; do not let upstream silently discard it')
            pred.append(dict(points=points))
        result = evaluator.evaluate_image(parse_gt(gt_files[key]), pred)
        results.append(result)
        per_image.append(dict(sample_id=key, gt=result['gtCare'], predictions=result['detCare'],
                              matched=result['detMatched']))
    return dict(evaluation_type='official_checkpoint_external_benchmark_validation',
                benchmark='ICDAR2015 test', images=len(rows),
                measured=evaluator.combine_results(results),
                counts=dict(gt=sum(r['gt'] for r in per_image),
                            predictions=sum(r['predictions'] for r in per_image),
                            matched=sum(r['matched'] for r in per_image)),
                exact_paddle_83_8_reproduction='NOT VERIFIED', evaluator_sha256=sha256(EVALUATOR)), per_image
