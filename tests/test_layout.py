"""Verify relocated entrypoints, isolated Monkey runtime and unchanged benchmark contract."""
import contextlib
import io
import json
import subprocess
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from cs431.common import ROOT, module, read_json, write_jsonl
from test_workflow import test_temp

monkey = module(ROOT/'monkey/run.py', 'monkey_workspace_entrypoint')


class LayoutTests(unittest.TestCase):
    def test_model_folders_reports_and_editor_workspace_exist(self):
        for name in ['monkey/phase1/report/official_metrics.json',
                     'monkey/phase2/report/metrics.json', 'monkey/phase3/report/REVIEW_job7957.md',
                     'paddle/configs/paddle_v5.json', 'paddle/vendor/eval_det_iou.py',
                     'paddle/tools/setup_paddle.py', 'paddle/phase2/report/README.md',
                     'comparison/README.md']:
            self.assertTrue((ROOT/name).is_file(), name)
        workspace = read_json(ROOT/'Monkey.code-workspace')
        self.assertEqual({f['path'] for f in workspace['folders']}, {'monkey','shared'})
        self.assertTrue(all((ROOT/f['path']).is_dir() for f in workspace['folders']))

    def test_paddle_tools_find_root_after_move(self):
        for name in ['setup_paddle', 'prepare_paddle_model']:
            tool = module(ROOT/f'paddle/tools/{name}.py', 'relocated_'+name)
            self.assertEqual(tool.ROOT, ROOT)

    def test_monkey_runtime_rejects_paddle_and_legacy_state(self):
        for path in [ROOT/'paddle/models', ROOT/'outputs/paddle', ROOT/'shared',
                     ROOT.parent/'_workspace_monkeyocr_text_detection_phase3']:
            with self.assertRaises(ValueError):
                monkey.safe_runtime(path)
        self.assertEqual(monkey.safe_runtime(ROOT/'monkey/.runtime/new'), ROOT/'monkey/.runtime/new')

    def test_root_dispatcher_status_and_real_phase3_plan_without_execution(self):
        runtime = ROOT/'monkey/.runtime'/('plan_only_'+uuid.uuid4().hex)
        result = subprocess.run([sys.executable, str(ROOT/'run.py'), 'monkey',
                                 '--runtime-root', str(runtime), 'status'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        status = json.loads(result.stdout)
        self.assertEqual(status['workspace'], str(ROOT/'monkey'))
        self.assertEqual(status['reports']['phase2'], str(ROOT/'monkey/phase2/report'))
        result = subprocess.run([sys.executable, str(ROOT/'run.py'), 'monkey',
                                 '--runtime-root', str(runtime), 'phase3', '--plan-only'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['workspace'], str(runtime/'phase3'))
        self.assertEqual(plan['configuration']['scheduler']['partition'], 'normal')
        self.assertFalse(runtime.exists(), 'plan/status must not create runtime or submit a job')

    def test_workspace_phase2_rejects_paddle_predictions(self):
        with test_temp() as tmp:
            pred = Path(tmp)/'wrong_model.jsonl'
            write_jsonl(pred, [dict(model='paddle')])
            args = types.SimpleNamespace(action='evaluate', predictions=str(pred))
            with patch.object(monkey, 'unpack', return_value=Path(tmp)/'frozen.jsonl'):
                with self.assertRaisesRegex(RuntimeError, 'original Monkey predictions'):
                    monkey.phase2_command(args, Path(tmp), sys.executable)

    def test_phase1_only_notebook_stops_before_phase2(self):
        folder = ROOT/'paddle/phase1/notebooks'
        first = read_json(folder/'official_eval.ipynb')
        full = read_json(folder/'pipeline_phase1_phase2.ipynb')
        self.assertLess(len(first['cells']), len(full['cells']))
        code = '\n'.join(''.join(c['source']) for c in first['cells'] if c['cell_type']=='code')
        self.assertIn("'phase1'", code)
        self.assertNotIn("'phase2'", code)
        self.assertIn('REVIEW_NOTES', code)


if __name__ == '__main__':
    unittest.main()
