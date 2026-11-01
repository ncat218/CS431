"""Fetch the pinned PaddleOCR repo (unmodified) and the official training checkpoint.

Layout of --out (the `--model-dir` of run.py):
  PaddleOCR/                                         git clone, tag v3.3.3, exact commit, clean tree
  weights/PP-OCRv5_server_det_pretrained.pdparams    official checkpoint, SHA256-verified
Same code/config/weights as the Phase 1 official tools/eval.py run.
"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT/'paddle/models/PP-OCRv5_server_det'))
    args = p.parse_args()
    cfg = json.loads((ROOT/'paddle/configs/paddle_v5.json').read_text(encoding='utf-8'))
    folder = Path(args.out)
    repo = folder/'PaddleOCR'
    if not (repo/'.git').is_dir():
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', '--depth', '1', '--branch', cfg['paddleocr_tag'],
                        cfg['paddleocr_repository'], str(repo)], check=True)
    if git(repo, 'rev-parse', 'HEAD') != cfg['paddleocr_commit']:
        raise RuntimeError('PaddleOCR clone is not the pinned commit; use a fresh --out')
    if git(repo, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('PaddleOCR clone has modified tracked files; use a fresh --out')
    if file_sha256(repo/cfg['config_relpath']) != cfg['config_sha256']:
        raise RuntimeError('Official config bytes differ from pin (check core.autocrlf)')
    if file_sha256(repo/cfg['upstream_iou_evaluator_relpath']) != file_sha256(ROOT/'paddle/vendor/eval_det_iou.py'):
        raise RuntimeError('Vendored ICDAR evaluator differs from the pinned PaddleOCR evaluator')
    print('VERIFIED PaddleOCR', cfg['paddleocr_tag'], cfg['paddleocr_commit'])
    dest = folder/'weights'/cfg['weights_file']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if file_sha256(dest) != cfg['weights_sha256']:
            raise RuntimeError('Existing weights have wrong hash; use a fresh model directory: '+str(dest))
        print('VERIFIED', dest.name)
        return
    temporary = dest.with_suffix(dest.suffix+'.download')
    digest = hashlib.sha256()
    with urllib.request.urlopen(cfg['weights_url'], timeout=120) as src, temporary.open('wb') as dst:
        while chunk := src.read(1024*1024):
            digest.update(chunk)
            dst.write(chunk)
    if digest.hexdigest() != cfg['weights_sha256']:
        raise RuntimeError('Downloaded official checkpoint hash mismatch')
    temporary.replace(dest)
    print('DOWNLOADED & VERIFIED', dest.name)


if __name__ == '__main__':
    main()
