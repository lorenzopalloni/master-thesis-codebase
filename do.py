import os
import argparse
import subprocess
from pathlib import Path

def check_venv():
    if not 'VIRTUAL_ENV' in os.environ.keys():
        raise Exception('Environmental variable VIRTUAL_ENV is not defined.')
    if not Path(os.environ['VIRTUAL_ENV']).name == 'binarization':
        subprocess.run('source ~/.venv/binarization/bin/activate') 

def do_test():
    check_venv()
    subprocess.run('python -m pytest tests -vv'.split(' '))

def do_coverage():
    check_venv()
    subprocess.run('coverage run -m pytest tests'.split(' '))
    subprocess.run('coverage report -m'.split(' '))

def main():
    parser = argparse.ArgumentParser(
        prog='dopy',
        description='Run custom configurations.'
    )
    parser.add_argument(
        'command',
        choices=[
            'test',
            'coverage'
        ],
    )
    args = parser.parse_args()

    command_dict = {
        'test': do_test,
        'coverage': do_coverage,
    }
    command_dict[args.command]()

if __name__ == '__main__':
    main()

