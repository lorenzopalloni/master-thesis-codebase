import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Union


def compress(
    input_fn: Union[Path, str],
    output_fn: Union[Path, str],
    crf: int = 23,
    scale_factor: int = 2,
):
    """Compresses a video.

    Args:
        input_fn (Union[Path, str]): Filename of the input video.
        output_fn (Union[Path, str]): Filename of the compressed video.
        crf (int, optional): Constant Rate Factor. Defaults to 23.
        scale_factor (int): Scale factor. Defaults to 2.
    """
    cmd = [
        'ffmpeg',
        '-i',
        f'{input_fn}',
        '-c:v',  # codec video
        'libx265',  # H.265/HEVC
        '-crf',  # constant rate factor
        f'{crf}',  # defaults to 23
        '-preset',  # faster -> less quality, slower -> better quality
        'medium',  # defaults to medium
        '-c:a',  # codec audio
        'aac',  # AAC audio format
        '-b:a',  # bitrate audio
        '128k',  # AAC audio at 128 kBit/s
        '-movflags',  # weird option
        'faststart',
        '-vf',  # video filters
        (
            f'scale=iw/{scale_factor}:ih/{scale_factor}'  # downscale
            ',format=yuv420p'  # output format, defaults to yuv420p
        ),
        # (
        #     f'scale=-{scale_factor}:iw'  # downscale
        #     ',format=yuv420p'  # output format, defaults to yuv420p
        # ),
        f'{output_fn}',
    ]
    subprocess.run(cmd, check=True)


def video_to_frames(
    input_fn: Union[Path, str],
    output_dir: Union[Path, str],
):
    """Splits a video into .jpg frames.

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
        f'{ (Path(output_dir) / Path(input_fn).stem).as_posix() }_%4d.jpg',
    ]
    subprocess.run(cmd, check=True)


def all_files_have_the_same_extension(folder: Union[Path, str]) -> bool:
    """Returns True if all the files in `folder` have the same extension."""
    folder = Path(folder)
    files = list(x for x in folder.iterdir() if not x.is_dir())
    return len(files) == 0 or len(set(x.suffix for x in files)) == 1


def assure_same_extension_among_files(
    folder: Union[Path, str]
) -> Union[bool, Exception]:
    """Returns True if all the files in `folder` have the same extension,
    otherwise raise an exception.
    """
    folder = Path(folder)
    if all_files_have_the_same_extension(folder):
        return True
    raise Exception(
        f'All files in "{folder.resolve().as_posix()}" must have the'
        ' same extension.'
    )


def prepare_original_videos_dir(
    input_dir: Union[Path, str],
    original_videos_namedir: str = 'original_videos',
) -> Path:
    """Assures a standard directory structure for original videos."""
    original_videos_namedir = 'original_videos'
    input_dir = Path(input_dir)

    case1 = input_dir.resolve().name != original_videos_namedir
    case11 = case1 and (input_dir / original_videos_namedir).is_dir()
    case2 = input_dir.resolve().name == original_videos_namedir

    if case11:
        original_videos_dir = input_dir / original_videos_namedir
        assure_same_extension_among_files(original_videos_dir)
    elif case1:
        assure_same_extension_among_files(input_dir)
        (original_videos_dir := input_dir / original_videos_namedir).mkdir()
        # mv <input_dir>/* -t <input_dir>/<original_videos_namedir>/
        for video_fp in input_dir.iterdir():
            if video_fp != original_videos_dir:
                shutil.move(
                    video_fp.as_posix(), original_videos_dir.as_posix()
                )
    elif case2:
        assure_same_extension_among_files(input_dir)
        original_videos_dir = input_dir
        if not len(list(original_videos_dir.parent.iterdir())) == 1:
            raise Exception(
                f'"{original_videos_dir.parent.resolve().as_posix()}" is expected to'
                f' contain only `./{original_videos_namedir}`, without any '
                'other files.'
            )
    return original_videos_dir


def prepare_directories(
    input_dir: Union[Path, str],
    original_videos_namedir: str = 'original_videos',
) -> Tuple[Path, Path, Path, Path]:
    """Assures a standard dir structure for original/compressed videos/frames."""
    original_dir = prepare_original_videos_dir(
        input_dir=input_dir, original_videos_namedir=original_videos_namedir
    )

    root_dir = original_dir.parent
    (compressed_videos_dir := root_dir / 'compressed_videos').mkdir(
        exist_ok=True
    )

    (original_frames_dir := root_dir / 'original_frames').mkdir(exist_ok=True)
    (compressed_frames_dir := root_dir / 'compressed_frames').mkdir(
        exist_ok=True
    )
    return (
        original_dir,
        compressed_videos_dir,
        original_frames_dir,
        compressed_frames_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='.')
    parser.add_argument('-s', '--scale_factor', type=int, default=2)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    (
        original_dir,
        compressed_videos_dir,
        original_frames_dir,
        compressed_frames_dir,
    ) = prepare_directories(input_dir)

    for original_fn in original_dir.iterdir():
        compressed_fn = Path(compressed_videos_dir, original_fn.stem + '.mp4')

        compress(
            input_fn=original_fn,
            output_fn=compressed_fn,
            scale_factor=args.scale_factor,
        )

        original_frames_subdir = original_frames_dir / original_fn.stem
        original_frames_subdir.mkdir(exist_ok=True)
        compressed_frames_subdir = compressed_frames_dir / compressed_fn.stem
        compressed_frames_subdir.mkdir(exist_ok=True)

        video_to_frames(
            input_fn=original_fn, output_dir=original_frames_subdir
        )
        video_to_frames(
            input_fn=compressed_fn, output_dir=compressed_frames_subdir
        )


if __name__ == '__main__':
    main()
