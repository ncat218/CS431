"""Try the original OOM image on CPU before committing to a new full run."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cs431.common import identity, sha256
from cs431.cpu_fallback import retry_cpu
from cs431.inference import model_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-dir', default=str(ROOT/'paddle/models/PP-OCRv5_server_det'))
    args = p.parse_args()
    image = Path(args.image).resolve()
    task = dict(sample_id='cpu_memory_probe', path=str(image), image_sha256=sha256(image))
    mid = identity(model_identity(args.model_dir))
    polygons, scores = retry_cpu(task, args.model_dir, mid, Path(args.out)/'probe',
                                 'explicit CPU memory probe; not scored or used to open Phase 2')
    print(f'CPU_PROBE_OK: {len(polygons)} boxes; original pixels and fixed model configuration preserved', flush=True)


if __name__ == '__main__':
    main()
