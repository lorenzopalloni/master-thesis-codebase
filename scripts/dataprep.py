"""Video preprocessing script that prepares original/compressed frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from binarization.videotools import (
    compress_video,
    prepare_directories,
    video_to_frames,
)


def arg_parse() -> argparse.Namespace:
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='.')
    parser.add_argument('-s', '--scale_factor', type=int, default=4)
    return parser.parse_args()


def main():
    """Main routine for video preprocessing."""
    args = arg_parse()
    input_dir = Path(args.input_dir)
    (
        original_video_dir,
        compressed_video_dir,
        original_frames_dir,
        compressed_frames_dir,
    ) = prepare_directories(input_dir)

    for original_video_path in original_video_dir.iterdir():
        compressed_video_path = Path(
            compressed_video_dir, original_video_path.stem + '.mp4'
        )

        compress_video(
            original_video_path=original_video_path,
            compressed_video_path=compressed_video_path,
            scale_factor=args.scale_factor,
        )

        original_frames_subdir = Path(
            original_frames_dir, original_video_path.stem
        )
        original_frames_subdir.mkdir(exist_ok=True)
        compressed_frames_subdir = Path(
            compressed_frames_dir, compressed_video_path.stem
        )
        compressed_frames_subdir.mkdir(exist_ok=True)

        video_to_frames(
            video_path=original_video_path,
            frames_dir=original_frames_subdir,
            ext=".png",
        )
        video_to_frames(
            video_path=compressed_video_path,
            frames_dir=compressed_frames_subdir,
            ext=".jpg",
        )


if __name__ == '__main__':
    main()
