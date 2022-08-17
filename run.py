import argparse
import subprocess
from pathlib import Path


def set_up_test_assets():
    current_dir = Path()
    tests_dir = current_dir / 'tests'
    assets_dir = tests_dir / 'assets'
    original_dir = assets_dir / 'original'
    encoded_dir = assets_dir / 'encoded'
    if not encoded_dir.exists():
        subprocess.run(
            'python ./binarization/video_preprocessing.py -i '
            f'{original_dir.as_posix()}'.split(' '),
            check=True,
        )


def run_test():
    set_up_test_assets()
    proc = subprocess.run('python -m pytest tests -vv'.split(' '), check=True)
    return proc.stdout


def run_coverage():
    set_up_test_assets()
    subprocess.run('coverage run -m pytest tests'.split(' '), check=True)
    subprocess.run('coverage report -m'.split(' '), check=True)


def main():
    parser = argparse.ArgumentParser(
        prog='python run.py', description='Run custom configurations.'
    )
    parser.add_argument(
        'command',
        choices=['test', 'coverage'],
    )
    args = parser.parse_args()

    command_dict = {
        'test': run_test,
        'coverage': run_coverage,
    }
    command_dict[args.command]()


if __name__ == '__main__':
    main()
