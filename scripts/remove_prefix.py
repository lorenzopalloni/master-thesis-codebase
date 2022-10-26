import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Usage: remove_prefix.py <target_dir> <prefix>')
    target_dir = Path(sys.argv[1]).resolve()
    print(f'{target_dir=}')
    prefix = sys.argv[2]
    print(f'{prefix=}')

    for path in target_dir.iterdir():
        new_name = path.name.replace(prefix, '')
        path.rename(path.parent / new_name)
