"""Create an isolated Linux GPU environment without modifying Kaggle/Colab base.

Stack = Phase 1 official_eval (Triệu, 31/10/2026): paddlepaddle-gpu 3.3.1 (cu126) +
PaddleOCR v3.3.3 requirements pinned in paddle/requirements.txt. Kaggle Python 3.13 works.
The venv isolates Paddle's nvidia-* wheels from Kaggle's torch (NCCL clash seen in Phase 1).
"""
import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PADDLE = 'paddlepaddle-gpu==3.3.1'


def setup_env():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def ensure_pip(python, env):
    """Repair a partially created venv without deleting packages or run artifacts."""
    probe = [python, '-I', '-m', 'pip', '--version']
    if subprocess.run(probe, env=env, capture_output=True).returncode == 0:
        return
    print('Bootstrapping pip in the Paddle environment', flush=True)
    subprocess.run([python, '-I', '-m', 'ensurepip', '--upgrade'], env=env)
    if subprocess.run(probe, env=env, capture_output=True).returncode != 0:
        # Linux distributions may omit ensurepip. Host pip can manage a pip-less
        # target interpreter without installing Paddle into the notebook kernel.
        subprocess.run([sys.executable, '-m', 'pip', '--python', python,
                        'install', 'pip==25.1.1'], env=env, check=True)
    subprocess.run(probe, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--venv', default=str(ROOT/'paddle/.venv'))
    p.add_argument('--cuda', choices=['cu126', 'cu118'], default='cu126')
    args = p.parse_args()
    if sys.platform != 'linux' or not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit('Use Kaggle/Colab Linux with Python 3.11-3.13. Do not install GPU Paddle into Windows/base Python.')
    subprocess.run(['nvidia-smi'], check=True)
    folder = Path(args.venv).resolve()
    if not (folder/'bin/python').exists():
        venv.EnvBuilder(with_pip=False).create(folder)
    python = str(folder/'bin/python')
    env = setup_env()
    ensure_pip(python, env)
    subprocess.run([python, '-I', '-m', 'pip', 'install', 'pip==25.1.1'], env=env, check=True)
    subprocess.run([python, '-I', '-m', 'pip', 'install', PADDLE,
                    '-i', f'https://www.paddlepaddle.org.cn/packages/stable/{args.cuda}/'], env=env, check=True)
    subprocess.run([python, '-I', '-m', 'pip', 'install', '-r', str(ROOT/'paddle/requirements.txt')], env=env, check=True)
    subprocess.run([python, '-I', '-m', 'pip', 'check'], env=env, check=True)
    subprocess.run([python, '-c', 'import paddle; assert paddle.is_compiled_with_cuda(); '
                    'assert paddle.device.cuda.device_count()>0; '
                    'paddle.set_device("gpu:0"); x=paddle.ones([8,8]); '
                    'print(paddle.matmul(x,x).numpy()); paddle.device.cuda.synchronize(); '
                    'print(paddle.__version__,paddle.device.cuda.get_device_name(0))'], env=env, check=True)
    print('Environment ready:', python)
    print('Next:', python, 'paddle/tools/prepare_paddle_model.py')


if __name__ == '__main__':
    main()
