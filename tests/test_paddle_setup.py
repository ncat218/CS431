import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1]/'paddle/tools/setup_paddle.py'
SPEC = importlib.util.spec_from_file_location('paddle_setup', PATH)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class PaddleSetupTests(unittest.TestCase):
    def test_repairs_real_environment_created_without_pip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'env'
            venv.EnvBuilder(with_pip=False).create(folder)
            python = str(folder/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
            env = setup.setup_env()
            before = subprocess.run([python, '-I', '-m', 'pip', '--version'],
                                    env=env, capture_output=True)
            self.assertNotEqual(before.returncode, 0)
            setup.ensure_pip(python, env)
            subprocess.run([python, '-I', '-m', 'pip', '--version'], env=env, check=True)
            with patch.object(setup.subprocess, 'run', wraps=subprocess.run) as run:
                setup.ensure_pip(python, env)
                self.assertEqual(run.call_count, 1)

    def test_missing_ensurepip_uses_host_pip_target_not_host_install(self):
        failed = subprocess.CompletedProcess([], 1)
        passed = subprocess.CompletedProcess([], 0)
        with patch.object(setup.subprocess, 'run', side_effect=[failed, failed, failed, passed, passed]) as run:
            setup.ensure_pip('/target/bin/python', {})
        command = run.call_args_list[3].args[0]
        self.assertEqual(command[1:], ['-m', 'pip', '--python', '/target/bin/python',
                                      'install', 'pip==25.1.1'])

    def test_notebook_python_paths_do_not_leak_into_setup(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/host/hooks', 'PYTHONHOME': '/host/python'}):
            env = setup.setup_env()
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
