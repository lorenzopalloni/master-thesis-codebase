# pylint: disable=line-too-long
"""Collection of scripts for general package-level utilities"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable
from pathlib import Path


def check_venv(venv_name: str):
    """Raises an exception if a specific virtual env is not currently active"""
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if not virtual_env or venv_name not in virtual_env:
        raise ValueError(f'You should enable `{venv_name}` virtual env.')


def prepare_tests(project_dir: Path | None = None):
    """Sets up resources for testing"""
    if project_dir is None:
        project_dir = Path(__file__).parent.parent
    tests_dir = project_dir / 'tests'
    assets_dir = tests_dir / 'assets'
    original_videos_dir = assets_dir / 'original_videos'
    compressed_videos_dir = assets_dir / 'compressed_videos'
    script_fp = Path(project_dir, 'scripts', 'video_preprocessing.py')
    if not compressed_videos_dir.exists():
        subprocess.run(
            f'python {script_fp.as_posix()} -i '
            f'{original_videos_dir.as_posix()}'.split(' '),
            check=True,
        )


def install_requirements():
    """Installs requirements assuming `binarization` as the active venv name"""
    venv_name = 'binarization'
    check_venv(venv_name)

    command_list = [
        "pip install -U pip",
        "pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113",
        "pip install piq lpips",  # metrics for image quality assessment
        "pip install pytest pylint mypy black flake8 pre-commit",  # development packages
        "pip install mlflow",  # tracking experiments
        "pip install numpy matplotlib seaborn pandas",
        # f"pip install -e {os.path.join('..', 'gifnoc')}",  # I'll probably hardcode it in binarization
    ]
    for cmd in command_list:
        subprocess.run(cmd.split(' '), check=True)


def run_tests():
    """Sets up resources for testing, and runs tests"""
    prepare_tests()
    proc = subprocess.run('python -m pytest tests -vv'.split(' '), check=True)
    return proc.stdout


def run_coverage():
    """Sets up resources for testing, runs tests, and runs coverage"""
    prepare_tests()
    subprocess.run('coverage run -m pytest tests'.split(' '), check=True)
    subprocess.run('coverage report -m'.split(' '), check=True)


def parse_args(command_dict: dict[str, Callable]) -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog=f'python {__file__}', description='Run custom configurations.'
    )
    parser.add_argument('command', choices=command_dict.keys())
    return parser.parse_args()


def main():
    """Run tests, coverage, or install some required packages."""
    command_dict = {
        'tests': run_tests,
        'coverage': run_coverage,
        'install_requirements': install_requirements,
    }
    args = parse_args(command_dict)
    command_dict[args.command]()


if __name__ == '__main__':
    main()
