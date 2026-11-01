import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from . import gate, icdar, inference, manifest
from .common import ROOT, identity, jsonl, module, now, read_json, save_json, sha256, write_jsonl


def overlays(tasks, predictions, gt_files, output, count=20):
    pred = {r['sample_id']: r for r in jsonl(predictions)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for task in tasks[:count]:
        with Image.open(task['path']) as source:
            im = source.convert('RGB')
        draw = ImageDraw.Draw(im)
        for entry in icdar.parse_gt(gt_files[task['sample_id']]):
            points = entry['points']
            draw.line(points + points[:1], fill='gray' if entry['ignore'] else 'lime', width=2)
        for coords in pred[task['sample_id']]['polygons']:
            points = list(zip(coords[::2], coords[1::2]))
            draw.line(points + points[:1], fill='red', width=2)
        im.save(output/(task['sample_id']+'.jpg'))
    (output/'README.txt').write_text('Green=valid GT; gray=ignored GT; red=prediction. Inspect positioning, missing text and false positives.\n')


def phase1(args):
    model_id = inference.model_identity(args.model_dir)
    tasks, inventory, gt_files = icdar.dataset(args.images, args.gt)
    inventory['source_description'] = args.dataset_source
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cid = gate.clone_id()
    # Bind even reusable inference to this clone and dataset. Old outputs cannot unlock a new clone.
    run_hash = identity(dict(dataset=inventory, clone_id=cid))
    predictions = inference.launch(tasks, out/'inference', args.model_dir, model_id, run_hash, args.gpus)
    metrics, per_image = icdar.evaluate(predictions, gt_files)
    save_json(out/'dataset_inventory.json', inventory)
    save_json(out/'config_snapshot.json', read_json(inference.CONFIG))
    save_json(out/'model_identity.json', model_id)
    save_json(out/'metrics.json', metrics)
    write_jsonl(out/'per_image.jsonl', per_image)
    overlays(tasks, predictions, gt_files, out/'overlays')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out/'pip_freeze.txt').write_text(freeze, encoding='utf-8')
    p, r, h = (100*metrics['measured'][k] for k in ('precision', 'recall', 'hmean'))
    ref = read_json(inference.CONFIG)['phase1_official_reference']
    rp, rr, rh = (100*ref[k] for k in ('precision', 'recall', 'hmean'))
    (out/'REVIEW_ME.md').write_text(
        f'# Paddle Phase 1 — requires review\n\nICDAR2015 test: 500 images.\n\n'
        f'Precision {p:.3f}%; Recall {r:.3f}%; Hmean {h:.3f}%.\n\n'
        f'Reference with the same code/config/weights/packages (official tools/eval.py run, Phase 1 report): '
        f'P {rp:.3f}% / R {rr:.3f}% / Hmean {rh:.3f}% (delta Hmean {h-rh:+.3f} pp). A non-trivial delta means '
        'this adapter does not reproduce the official pipeline; investigate before review. '
        'Diagnostic only, not an automatic pass threshold.\n\n'
        'Official PaddleOCR evaluation pipeline on a public benchmark. Exact Paddle 83.8% reproduction: NOT VERIFIED.\n\n'
        'Inspect metrics.json, dataset provenance/inventory, per_image.jsonl, worker logs and overlays/. '
        'No published ICDAR2015 target is asserted for this PP-OCRv5 checkpoint; acceptance requires '
        'a written assessment, not comparison with Monkey or the unrelated 83.8% figure. '
        'The program does not automatically accept the result.\n', encoding='utf-8')
    artifacts = {str(p.relative_to(out)).replace('\\', '/'): sha256(p)
                 for p in out.rglob('*') if p.is_file() and p.name not in {'phase1_result.json', '.controller.lock'}}
    report = dict(status='EVALUATED_REQUIRES_REVIEW', clone_id=cid, completed_utc=now(),
                  benchmark='ICDAR2015 test', images=500, model_identity=model_id, artifacts=artifacts)
    save_json(out/'phase1_result.json', report)
    print((out/'REVIEW_ME.md').read_text(encoding='utf-8'))
    print('Phase 2 remains blocked until the operator inspects these artifacts and runs paddle review.')


def phase2(args):
    gate.require_phase1()  # Fail before loading frameworks/model/datasets on a fresh clone.
    model_id = inference.model_identity(args.model_dir)
    receipt = gate.require_phase1(model_id)
    frozen = manifest.unpack()
    lock = manifest.verify_shared(frozen)
    locations = read_json(args.locations)
    tasks = manifest.build_tasks(frozen, locations)
    if len(tasks) != lock['sources'] + lock['variants']:
        raise RuntimeError('Unexpected shared task count')
    missing = [t['sample_id'] for t in tasks if not Path(t['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not found; correct local roots, e.g. {missing[:3]}')
    out = Path(args.out).resolve()
    predictions = inference.launch(tasks, out/'inference', args.model_dir, model_id,
                                   lock['manifest_sha256'], args.gpus)
    evaluator = module(ROOT/'shared/evaluate_shared.py', 'frozen_phase2_evaluator')
    result = evaluator.evaluate(frozen, predictions, ROOT/'shared/frozen/evaluation_protocol.json',
                                out/'metrics', bootstrap=1000)
    save_json(out/'phase2_provenance.json', dict(
        phase1_review=receipt, shared_lock=lock, model_identity=model_id,
        hardware={p.parent.name:read_json(p)['gpu'] for p in (out/'inference').glob('gpu*/identity.json')},
        predictions_sha256=sha256(predictions), completed_utc=now()))
    # GT overlays are produced only after inference; workers never receive GT.
    subprocess.run([sys.executable, str(ROOT/'shared/visualize_pairs.py'),
                    '--manifest', str(frozen), '--locations', str(args.locations),
                    '--predictions', str(predictions), '--output-dir', str(out/'overlays'),
                    '--worst', str(out/'metrics/worst_recall_sources.json'), '--limit', '12'], check=True)
    print('Complete:', out/'metrics/clean_vs_degraded.md')
    return result


def compare(paddle_metrics, monkey_metrics, output):
    lock = manifest.verify_shared()
    results = [read_json(paddle_metrics), read_json(monkey_metrics)]
    if [r['model'] for r in results] != ['paddle', 'monkey']:
        raise ValueError('Expected Paddle and original Monkey baseline metrics, in that order')
    for r in results:
        for field in ('manifest_sha256', 'protocol_sha256', 'evaluator_sha256'):
            if r[field] != lock[field]:
                raise ValueError('Cannot compare different shared artifacts: ' + field)
        if r['unique_sources'] != lock['sources'] or r['degraded_variants'] != lock['variants']:
            raise ValueError('Cannot compare different source/variant counts')
    table = ['# Phase 2: shared full-master baseline', '',
             '| Model | Condition | Sources | Variants | P (%) | R (%) | Hmean (%) | Coverage (%) | Mean latency (ms) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for condition, field in [('clean', 'clean_unique'), ('degraded', 'degraded_instance')]:
            m = result[field]
            values = ' | '.join(f'{100*m[k]:.3f}' for k in ('precision','recall','hmean','word_coverage_recall'))
            latency=result.get('latency_with_decode_and_postprocess',{}).get(condition,{})
            ms=f'{latency["mean_ms"]:.2f}' if latency.get('available') else 'N/A'
            table.append(f'| {result["model"]} | {condition} | {result["unique_sources"]} | '
                         f'{result["degraded_variants"] if condition == "degraded" else 0} | {values} | {ms} |')
    table += ['', 'Gap = degraded − clean, percentage points; paired source-weighted micro.', '',
              '| Model | Metric | Gap (pp) | Paired bootstrap 95% CI (pp) |',
              '|---|---|---:|---|']
    for result in results:
        for metric,delta in result['paired_source_weighted_micro']['delta_pp'].items():
            ci=result.get('bootstrap',{}).get('delta_pp_95ci',{}).get(metric)
            interval=f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else 'N/A'
            table.append(f'| {result["model"]} | {metric} | {delta:+.3f} | {interval} |')
    table += ['', 'Clean is evaluated once per source. Coverage is diagnostic. Synthetic degradation; '
              'word-versus-line GT and prior pretraining exposure limit interpretation.', '',
              'Latency is recorded per run/hardware; do not claim an A100/T4/H200 speed ranking from these runs.', '',
              'Monkey baseline hardware: two Tesla T4 workers. Paddle hardware: see phase2_provenance.json '
              'and inference/gpu*/identity.json. Latency includes decode/preprocess/postprocess.', '',
              'Manifest SHA256: ' + lock['manifest_sha256']]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(table)+'\n', encoding='utf-8')
    return path
