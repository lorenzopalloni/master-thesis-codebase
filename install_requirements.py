import os
import subprocess
from pathlib import Path


def check_venv(venv_name: str):
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if not virtual_env or venv_name not in virtual_env:
        raise ValueError("You should enable `binarization` virtual env.")

def main():
    venv_name = 'binarization'
    check_venv(venv_name)

    command_list = [
        "pip install -U pip",
        "pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113",
        "pip install piq",
        "pip install lpips",
    ]
    for cmd in command_list:
        subprocess.run(cmd.split(' '), check=True)

if __name__ == '__main__':
    main()

