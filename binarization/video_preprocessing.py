from pathlib import Path
from typing import Union, Tuple
import argparse
import subprocess
import shutil


def compress(
    input_fn: Union[Path, str],
    output_fn: Union[Path, str],
    crf: int = 23,
):
    """Compress a video.

    Args:
        input_fn (Union[Path, str]): Filename of the input video.
        output_fn (Union[Path, str]): Filename of the compressed video.
        crf (int, optional): Constant Rate Factor. Defaults to 23.
    """
    cmd = [
        'ffmpeg',
        '-i',
        f'{input_fn}',
        '-c:v',  # codec video
        'libx265',  # H.265/HEVC
        '-crf',  # constant rate factor
        f'{crf}',  # the default is 28
        '-preset',  # faster -> less quality, slower -> better quality
        'medium',  # the default is medium
        '-c:a',  # codec audio
        'aac',  # AAC audio format
        '-b:a',  # bitrate audio
        '128k',  # AAC audio at 128 kBit/s
        '-movflags',  # weird option
        'faststart',
        '-vf',  # video filters
        'scale=iw/2:ih/2,format=yuv420p',  # scaling 1/2, formatting yuv420p
        # 'scale=-2:540,format=yuv420p',  # from compress_test_videos.sh
        f'{output_fn}',
    ]
    subprocess.run(cmd)


def video_to_frames(
    input_fn: Union[Path, str],
    output_dir: Union[Path, str],
):
    """Split a video into .jpg frames.

    Args:
        input_fn (Union[Path, str]): Filename of the input video.
        output_dir (Union[Path, str]): Output directory where all the frames
        will be stored.
    """
    cmd = [
        'ffmpeg',
        '-i',
        f'{input_fn}',
        '-vf',  # video filters
        r'select=not(mod(n\,1))',  # select all frames, ~same as 'select=1'
        '-vsync',
        'vfr',  # original option in fede-vaccaro/fast-sr-unet
        # '-vsync', '0',  # should avoid drops or duplications
        '-q:v',
        '1',
        f'{ output_dir / input_fn.stem }_%4d.jpg',
    ]
    subprocess.run(cmd)


def all_files_have_the_same_extension(folder: Union[Path, str]) -> bool:
    """Return True if all the files in `folder` have the same extension."""
    folder = Path(folder)
    files = list(x for x in folder.iterdir() if not x.is_dir())
    return len(files) == 0 or len(set(x.suffix for x in files)) == 1


def assure_same_extension_among_files(
    folder: Union[Path, str]
) -> Union[bool, Exception]:
    """Return True if all the files in `folder` have the same extension,
    otherwise raise an exception.
    """
    folder = Path(folder)
    if all_files_have_the_same_extension(folder):
        return True
    else:
        raise Exception(
            f'All files in "{folder.resolve().as_posix()}" must have the'
            ' same extension.'
        )


def prepare_original_dir(input_dir: Union[Path, str]) -> Path:
    input_dir = Path(input_dir)

    case1 = input_dir.resolve().name != 'original'
    case11 = case1 and (input_dir / 'original').is_dir()
    case2 = input_dir.resolve().name == 'original'

    if case11:
        original_dir = input_dir / 'original'
        assure_same_extension_among_files(original_dir)
    elif case1:
        assure_same_extension_among_files(input_dir)
        (original_dir := input_dir / 'original').mkdir()
        # move all files in <input_dir>/ to <input_dir>/original, excluding
        # the latter
        for f in input_dir.iterdir():
            if f != original_dir:
                shutil.move(f.as_posix(), original_dir.as_posix())
    elif case2:
        assure_same_extension_among_files(input_dir)
        original_dir = input_dir
        if not len(list(original_dir.parent.iterdir())) == 1:
            raise Exception(
                f'"{original_dir.parent.resolve().as_posix()}" is expected to'
                ' contain only `./original`, without any other files.'
            )
    return original_dir


def prepare_directories(
    input_dir: Union[Path, str]
) -> Tuple[Path, Path, Path, Path]:
    original_dir = prepare_original_dir(input_dir)

    root_dir = original_dir.parent
    (encoded_dir := root_dir / 'encoded').mkdir(exist_ok=True)

    (original_frames_dir := root_dir / 'original_frames').mkdir(exist_ok=True)
    (encoded_frames_dir := root_dir / 'encoded_frames').mkdir(exist_ok=True)
    return original_dir, encoded_dir, original_frames_dir, encoded_frames_dir


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='.')
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    (
        original_dir,
        encoded_dir,
        original_frames_dir,
        encoded_frames_dir,
    ) = prepare_directories(input_dir)

    for original_fn in original_dir.iterdir():
        encoded_fn = encoded_dir / ('encoded_' + original_fn.stem + '.mp4')
        compress(input_fn=original_fn, output_fn=encoded_fn)

        (
            original_frames_subdir := original_frames_dir / original_fn.stem
        ).mkdir(exist_ok=True)
        (encoded_frames_subdir := encoded_frames_dir / encoded_fn.stem).mkdir(
            exist_ok=True
        )

        video_to_frames(
            input_fn=original_fn, output_dir=original_frames_subdir
        )
        video_to_frames(input_fn=encoded_fn, output_dir=encoded_frames_subdir)


if __name__ == '__main__':
    main()
