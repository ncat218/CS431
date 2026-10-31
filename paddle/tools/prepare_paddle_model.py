"""Download only three official inference files from an immutable HF revision."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT/'paddle/models/PP-OCRv5_server_det'))
    args = p.parse_args()
    cfg = json.loads((ROOT/'paddle/configs/paddle_v5.json').read_text())
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in cfg['model_files_sha256'].items():
        dest = folder/name
        if dest.exists():
            with dest.open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError('Existing model file has wrong hash; use a fresh model directory: '+str(dest))
            print('VERIFIED', name)
            continue
        url = f'https://huggingface.co/{cfg["model_repository"]}/resolve/{cfg["model_revision"]}/{name}'
        temporary = dest.with_suffix(dest.suffix+'.download')
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as src, temporary.open('wb') as dst:
            while chunk := src.read(1024*1024):
                digest.update(chunk)
                dst.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError('Downloaded official model hash mismatch: '+name)
        temporary.replace(dest)
        print('DOWNLOADED & VERIFIED', name)


if __name__ == '__main__':
    main()
