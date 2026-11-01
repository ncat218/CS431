"""Create an isolated Linux GPU environment without modifying Kaggle/Colab base.

Stack = Phase 1 official_eval (Triệu, 31/10/2026): paddlepaddle-gpu 3.3.1 (cu126) +
PaddleOCR v3.3.3 requirements pinned in paddle/requirements.txt. Kaggle Python 3.13 works.
The venv isolates Paddle's nvidia-* wheels from Kaggle's torch (NCCL clash seen in Phase 1).
"""
import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PADDLE = 'paddlepaddle-gpu==3.3.1'


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
        venv.EnvBuilder(with_pip=True).create(folder)
    python = str(folder/'bin/python')
    subprocess.run([python, '-m', 'pip', 'install', 'pip==25.1.1'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', PADDLE,
                    '-i', f'https://www.paddlepaddle.org.cn/packages/stable/{args.cuda}/'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', '-r', str(ROOT/'paddle/requirements.txt')], check=True)
    subprocess.run([python, '-m', 'pip', 'check'], check=True)
    subprocess.run([python, '-c', 'import paddle; assert paddle.is_compiled_with_cuda(); '
                    'assert paddle.device.cuda.device_count()>0; '
                    'paddle.set_device("gpu:0"); x=paddle.ones([8,8]); '
                    'print(paddle.matmul(x,x).numpy()); paddle.device.cuda.synchronize(); '
                    'print(paddle.__version__,paddle.device.cuda.get_device_name(0))'], check=True)
    print('Environment ready:', python)
    print('Next:', python, 'paddle/tools/prepare_paddle_model.py')


if __name__ == '__main__':
    main()
