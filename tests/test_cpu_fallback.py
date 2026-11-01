import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cs431 import cpu_fallback as cpu
from cs431.common import identity, jsonl, save_json, sha256, write_jsonl
from cs431 import inference


class CpuFallbackTests(unittest.TestCase):
    def test_worker_warmup_oom_saves_cpu_provenance_and_resumes_without_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'image.jpg'
            image.write_bytes(b'fixture; fake detector does not decode')
            task = dict(sample_id='large-image', path=str(image), image_sha256=sha256(image))
            tasks = root/'tasks.jsonl'
            write_jsonl(tasks, [task])
            model = dict(config_sha256='cfg')
            fake_cuda = SimpleNamespace(device_count=lambda:1, get_device_name=lambda n:'Fake GPU',
                                        synchronize=lambda:None, empty_cache=lambda:None)
            fake_paddle = SimpleNamespace(is_compiled_with_cuda=lambda:True,
                                          device=SimpleNamespace(cuda=fake_cuda), __version__='test',
                                          version=SimpleNamespace(cuda=lambda:'test', cudnn=lambda:'test'))
            detector = SimpleNamespace(load_report={}, transforms=[], resize={},
                                       predict=Mock(side_effect=MemoryError('out of memory on GPU')))
            constructor = Mock(return_value=detector)
            with patch.dict('sys.modules', {'paddle':fake_paddle,
                                           'cs431.ppocr_detector':SimpleNamespace(OfficialDetector=constructor)}), \
                 patch.object(inference, 'model_identity', return_value=model), \
                 patch.object(inference.subprocess, 'check_output', return_value='Fake GPU'), \
                 patch.object(cpu, 'retry_cpu', return_value=([], [])) as retry:
                inference.worker(tasks, root/'gpu0', 'fake_model', identity(model), 'manifest')
                rows = list(jsonl(root/'gpu0/predictions.jsonl'))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['execution_device'], 'cpu')
                self.assertEqual(rows[0]['fallback_reason'], 'gpu_oom')
                retry.assert_called_once()
                inference.worker(tasks, root/'gpu0', 'fake_model', identity(model), 'manifest')
                retry.assert_called_once()
                constructor.assert_called_once()
                self.assertEqual(list(jsonl(root/'gpu0/predictions.jsonl')), rows)

    def test_gpu_success_does_not_call_cpu(self):
        fallback, release = Mock(), Mock()
        output = ([[0, 0, 1, 0, 1, 1, 0, 1]], [0.8])
        self.assertEqual(cpu.predict_or_cpu(lambda:output, fallback, release), (output, 'gpu', None))
        fallback.assert_not_called(); release.assert_not_called()

    def test_gpu_oom_releases_memory_and_returns_cpu_output(self):
        events = []
        def gpu():
            raise MemoryError('ResourceExhaustedError: Out of memory error on GPU 0')
        def fallback(reason):
            events.append('cpu')
            self.assertIn('Out of memory', reason)
            return [], []
        result, device, reason = cpu.predict_or_cpu(gpu, fallback, lambda:events.append('release'))
        self.assertEqual(events, ['release', 'cpu'])
        self.assertEqual((result, device), (([], []), 'cpu'))

    def test_other_errors_and_cpu_failure_stop_the_run(self):
        for error in (MemoryError('host out of memory'), RuntimeError('invalid CUDA input'),
                      ValueError('bad image')):
            with self.subTest(error=error):
                fallback = Mock()
                with self.assertRaises(type(error)):
                    cpu.predict_or_cpu(Mock(side_effect=error), fallback, Mock())
                fallback.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'CPU failed'):
            cpu.predict_or_cpu(Mock(side_effect=MemoryError('out of memory on GPU')),
                               Mock(side_effect=RuntimeError('CPU failed')), Mock())

    def test_cpu_lock_serializes_workers_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'slot'
            entered = threading.Event()
            def second():
                with cpu.cpu_slot(path):
                    entered.set()
            with self.assertRaises(ValueError):
                with cpu.cpu_slot(path):
                    thread = threading.Thread(target=second)
                    thread.start()
                    self.assertFalse(entered.wait(0.2))
                    raise ValueError('simulated failure')
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertTrue(entered.is_set())

    def test_subprocess_identity_and_explicit_cpu_device(self):
        with tempfile.TemporaryDirectory() as directory:
            task = dict(sample_id='sample', path='original.jpg', image_sha256='imagehash')
            def fake_run(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                dest = command[command.index('--out')+1]
                save_json(dest, dict(sample_id='sample', image_sha256='imagehash',
                                    model_identity_sha256='modelhash', polygons=[], scores=[]))
            with patch.object(cpu.subprocess, 'run', side_effect=fake_run):
                self.assertEqual(cpu.retry_cpu(task, 'model', 'modelhash',
                                               Path(directory)/'gpu0', 'gpu OOM'), ([], []))
            def wrong_run(command, **kwargs):
                dest = command[command.index('--out')+1]
                save_json(dest, dict(sample_id='another', image_sha256='imagehash',
                                    model_identity_sha256='modelhash', polygons=[], scores=[]))
            with patch.object(cpu.subprocess, 'run', side_effect=wrong_run):
                with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                    cpu.retry_cpu(task, 'model', 'modelhash', Path(directory)/'gpu0', 'gpu OOM')
