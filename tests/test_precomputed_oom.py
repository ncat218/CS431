"""Host-precomputed predictions for images that exceed T4 VRAM and Kaggle RAM (CPU fixtures only)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cs431 import cpu_fallback as cpu
from cs431.common import read_json, sha256
from cs431.inference import CONFIG

QUAD = [0., 0., 10., 0., 10., 10., 0., 10.]


def write_precomputed(root, image_sha, sample_id, **override):
    cfg = read_json(CONFIG)
    data = dict(detector='OfficialDetector cpu', environment=dict(note='fixture'),
                config_sha256=cfg['config_sha256'], weights_sha256=cfg['weights_sha256'],
                paddleocr_commit=cfg['paddleocr_commit'],
                entries={image_sha: dict(sample_id=sample_id, image_sha256=image_sha, polygons=[QUAD],
                                         scores=[0.9], execution_device='cpu_local_precomputed')})
    data.update(override)
    path = Path(root)/'oom.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


class PrecomputedOomTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root/'big.jpg'
        image.write_bytes(b'fixture image bytes')
        self.task = dict(sample_id='clean::BIG', path=str(image), image_sha256=sha256(image))

    def test_matching_image_uses_precomputed_and_records_provenance(self):
        path = write_precomputed(self.root, self.task['image_sha256'], 'clean::BIG')
        with patch.dict(os.environ, {cpu.PRECOMPUTED_ENV: str(path)}), \
             patch.object(cpu, 'retry_cpu') as retry:
            (polys, scores), device = cpu.oom_retry(self.task, 'model', 'idhash', self.root/'gpu0', 'gpu OOM')
        retry.assert_not_called()
        self.assertEqual((polys, scores, device), ([QUAD], [0.9], 'cpu_local_precomputed'))
        saved = list((self.root/'gpu0/cpu_fallback').glob('*.precomputed.json'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(read_json(saved[0])['gpu_oom'], 'gpu OOM')

    def test_other_image_falls_back_to_cpu_retry(self):
        path = write_precomputed(self.root, 'f'*64, 'clean::OTHER')
        with patch.dict(os.environ, {cpu.PRECOMPUTED_ENV: str(path)}), \
             patch.object(cpu, 'retry_cpu', return_value=([], [])) as retry:
            _, device = cpu.oom_retry(self.task, 'model', 'idhash', self.root/'gpu0', 'gpu OOM')
        retry.assert_called_once()
        self.assertEqual(device, 'cpu')

    def test_rejects_different_weights_or_sample(self):
        path = write_precomputed(self.root, self.task['image_sha256'], 'clean::BIG', weights_sha256='0'*64)
        with patch.dict(os.environ, {cpu.PRECOMPUTED_ENV: str(path)}):
            with self.assertRaisesRegex(RuntimeError, 'weights_sha256'):
                cpu.oom_retry(self.task, 'model', 'idhash', self.root/'gpu0', 'gpu OOM')
        path = write_precomputed(self.root, self.task['image_sha256'], 'deg::WRONG')
        with patch.dict(os.environ, {cpu.PRECOMPUTED_ENV: str(path)}):
            with self.assertRaisesRegex(RuntimeError, 'another sample'):
                cpu.oom_retry(self.task, 'model', 'idhash', self.root/'gpu0', 'gpu OOM')

    def test_identity_binds_file_and_is_absent_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cpu.PRECOMPUTED_ENV, None)
            self.assertIsNone(cpu.precomputed_identity())
        path = write_precomputed(self.root, self.task['image_sha256'], 'clean::BIG')
        with patch.dict(os.environ, {cpu.PRECOMPUTED_ENV: str(path)}):
            ident = cpu.precomputed_identity()
        self.assertEqual(ident['file_sha256'], sha256(path))
        self.assertEqual(ident['image_sha256'], [self.task['image_sha256']])


if __name__ == '__main__':
    unittest.main()
