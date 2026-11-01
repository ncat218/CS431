"""Diagnostic only: compare CPU predictions with saved GPU Phase 1 predictions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cs431.common import identity, jsonl, save_json
from cs431.cpu_fallback import retry_cpu
from cs431.inference import model_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--model-dir', default=str(ROOT/'paddle/models/PP-OCRv5_server_det'))
    p.add_argument('--out', required=True, help='Outside Phase 1 report; does not alter its evidence')
    args = p.parse_args()
    folder, output = Path(args.run)/'inference', Path(args.out)
    mid = identity(model_identity(args.model_dir))
    predictions = {r['sample_id']:r for r in jsonl(folder/'predictions.jsonl')}
    tasks = [t for f in sorted(folder.glob('tasks_gpu*.jsonl')) for t in jsonl(f)]
    # Fixed, label-independent sample, chosen before inspecting any results.
    tasks = sorted(tasks, key=lambda t:t['sample_id'])[:3]
    if len(tasks) != 3:
        raise RuntimeError('Need a completed Phase 1 run')
    comparisons = []
    for task in tasks:
        gpu = predictions[task['sample_id']]
        if gpu['model_identity_sha256'] != mid or gpu.get('execution_device') != 'gpu':
            raise RuntimeError('Expected current-model GPU reference predictions')
        polys, scores = retry_cpu(task, args.model_dir, mid, output/'diagnostics',
                                 'explicit CPU/GPU diagnostic, not an OOM')
        same_polys = sorted(polys) == sorted(gpu['polygons'])
        comparisons.append(dict(sample_id=task['sample_id'], gpu_boxes=len(gpu['polygons']),
                                cpu_boxes=len(polys), polygon_sets_equal=same_polys,
                                ordered_scores_equal=scores == gpu['scores']))
    result = dict(model_identity_sha256=mid, comparisons=comparisons,
                  all_polygon_sets_equal=all(r['polygon_sets_equal'] for r in comparisons),
                  scope='Three fixed Phase 1 images only; not proof of equivalence on all images or automatic approval')
    save_json(output/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
