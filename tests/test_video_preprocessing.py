# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import pytest

from scripts.video_preprocessing import (
    files_have_same_ext,
    prepare_directories,
    prepare_original_videos_dir,
)


class TestAllFilesHaveTheSameExtension:
    def test_when_false(self, tmp_path):
        d = tmp_path / 'original_videos'
        d.mkdir()
        (d / 'a.txt').touch()
        (d / 'b.mp4').touch()
        assert not files_have_same_ext(d)

    def test_when_empty(self, tmp_path):
        d = tmp_path / 'original_videos'
        d.mkdir()
        assert files_have_same_ext(d)

    def test_when_true(self, tmp_path):
        d = tmp_path / 'original_videos'
        d.mkdir()
        (d / 'a.mp4').touch()
        (d / 'b.mp4').touch()
        assert files_have_same_ext(d)


class TestPrepareOriginalDir:
    def test_when_extensions_differ(self, tmp_path):
        d = tmp_path / 'original_videos'
        d.mkdir()
        (d / 'a.mp4').touch()
        (d / 'b.txt').touch()
        original_dir = prepare_original_videos_dir(d)
        assert (original_dir / 'a.mp4').exists()
        assert (original_dir / 'b.txt').exists()

    def test_case1(self, tmp_path):
        d = tmp_path / 'data'
        d.mkdir()
        (d / 'a.mp4').touch()
        (d / 'b.mp4').touch()
        original_dir = prepare_original_videos_dir(d)
        assert not (d / 'a.mp4').exists()
        assert not (d / 'b.mp4').exists()
        assert (original_dir / 'a.mp4').exists()
        assert (original_dir / 'b.mp4').exists()

    def test_case11(self, tmp_path):
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        original_dir = data_dir / 'original_videos'
        original_dir.mkdir()
        (original_dir / 'a.mp4').touch()
        (original_dir / 'b.mp4').touch()
        original_dir = prepare_original_videos_dir(data_dir)
        assert not (data_dir / 'a.mp4').exists()
        assert not (data_dir / 'b.mp4').exists()
        assert (original_dir / 'a.mp4').exists()
        assert (original_dir / 'b.mp4').exists()

    def test_case2(self, tmp_path):
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        original_dir = data_dir / 'original_videos'
        original_dir.mkdir()
        (original_dir / 'a.mp4').touch()
        (original_dir / 'b.mp4').touch()
        original_dir = prepare_original_videos_dir(original_dir)
        assert not (data_dir / 'a.mp4').exists()
        assert not (data_dir / 'b.mp4').exists()
        assert (original_dir / 'a.mp4').exists()
        assert (original_dir / 'b.mp4').exists()

    def test_case2_with_exception(self, tmp_path):
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        original_dir = data_dir / 'original_videos'
        original_dir.mkdir()
        (original_dir / 'a.mp4').touch()
        (original_dir / 'b.mp4').touch()
        (data_dir / 'c.mp4').touch()
        with pytest.raises(Exception):
            prepare_original_videos_dir(original_dir)


def test_prepare_directories(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    (d / 'a.mp4').touch()
    (d / 'b.mp4').touch()
    (
        original_dir,
        compressed_videos_dir,
        original_frames_dir,
        encoded_frames_dir,
    ) = prepare_directories(d)
    del (
        original_dir,
        compressed_videos_dir,
        original_frames_dir,
        encoded_frames_dir,
    )
    assert (d / 'compressed_videos').is_dir()
    assert (d / 'original_frames').is_dir()
    assert (d / 'compressed_frames').is_dir()
