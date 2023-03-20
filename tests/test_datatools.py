# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import numpy as np
import pytest  # noqa: F401
import torchvision

from binarization import datatools


class TestGetStartingRandomPosition:
    def test_naive(self):
        height = 10
        patch_size = 4
        for i in range(10):
            np.random.seed(i)
            actual = datatools.get_starting_random_position(height, patch_size)
            assert 0 <= actual < 6

    def test_when_axis_lower_than_patch_size(self):
        height = 3
        patch_size = 4
        expected = 0
        actual = datatools.get_starting_random_position(height, patch_size)
        assert actual == expected


def test_compose():
    ds = torchvision.datasets.FakeData(
        size=2,
        image_size=(3, 16, 16),
        num_classes=2,
        random_offset=42,
    )
    it = iter(ds)
    x, _ = next(it)
    y, _ = next(it)
    torchvision_compose = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            lambda x: (x - 0.5) * 2.0,
        ]
    )
    expected = torchvision_compose(x)
    my_compose = datatools.compose(
        torchvision.transforms.ToTensor(),
        lambda x: (x - 0.5) * 2.0,
    )
    actual = my_compose(x)
    assert expected.equal(actual)


def test_compute_adjusted_dimension():
    assert datatools.make_4times_divisible(256) == 256
    assert datatools.make_4times_divisible(270) == 288
    assert datatools.make_4times_divisible(540) == 544


class TestListHaveSameElements:
    def test_true(self):
        list1 = list('abcd')
        list2 = list('dcab')
        expected = True
        actual = datatools.lists_have_same_elements(list1, list2)
        assert expected == actual

    def test_false(self):
        list1 = list('bcde')
        list2 = list('dcab')
        expected = False
        actual = datatools.lists_have_same_elements(list1, list2)
        assert expected == actual

    def test_false_different_lengths(self):
        list1 = list('bcd')
        list2 = list('dcab')
        expected = False
        actual = datatools.lists_have_same_elements(list1, list2)
        assert expected == actual

    def test_false_equal_lengths_v1(self):
        list1 = list('abcd')
        list2 = list('aabc')
        expected = False
        actual = datatools.lists_have_same_elements(list1, list2)
        assert expected == actual

    def test_false_equal_lengths_v2(self):
        list1 = list('aabc')
        list2 = list('abcd')
        expected = False
        actual = datatools.lists_have_same_elements(list1, list2)
        assert expected == actual


def test_list_directories(tmp_path):
    my_folder_dir = tmp_path / 'original_frames'
    video_a_dir = my_folder_dir / 'video_a'
    video_b_dir = my_folder_dir / 'video_b'

    my_folder_dir.mkdir()
    video_a_dir.mkdir()
    video_b_dir.mkdir()

    expected = [video_a_dir, video_b_dir]
    actual = datatools.list_directories(my_folder_dir)
    assert datatools.lists_have_same_elements(expected, actual)


class TestListFiles:
    def test_when_same_extension(self, tmp_path):
        my_folder_dir = tmp_path / 'my_folder'
        a_fn = my_folder_dir / 'a.jpg'
        b_fn = my_folder_dir / 'b.jpg'
        my_subfolder_dir = my_folder_dir / 'my_subfolder'
        c_fn = my_subfolder_dir / 'c.jpg'

        my_folder_dir.mkdir()
        a_fn.touch()
        b_fn.touch()
        my_subfolder_dir.mkdir()
        c_fn.touch()

        expected = [a_fn, b_fn]
        actual = datatools.list_files(my_folder_dir, '.jpg')
        assert datatools.lists_have_same_elements(expected, actual)

    def test_when_different_extentions(self, tmp_path):
        my_folder_dir = tmp_path / 'my_folder'
        a_fn = my_folder_dir / 'a.txt'
        b_fn = my_folder_dir / 'b.jpg'
        my_subfolder_dir = my_folder_dir / 'my_subfolder'
        c_fn = my_subfolder_dir / 'c.jpg'

        my_folder_dir.mkdir()
        a_fn.touch()
        b_fn.touch()
        my_subfolder_dir.mkdir()
        c_fn.touch()

        expected = [b_fn]
        with pytest.warns(UserWarning):
            actual = datatools.list_files(my_folder_dir, '.jpg')
        assert datatools.lists_have_same_elements(expected, actual)

    def test_lexicographic(self, tmp_path):
        my_folder_dir = tmp_path / 'my_folder'
        fns = [
            my_folder_dir / name
            for name in [
                'a_0002.jpg',
                'a_0004.jpg',
                'a_0001.jpg',
                'a_0003.jpg',
            ]
        ]
        my_folder_dir.mkdir()
        for fn in fns:
            fn.touch()

        expected = [
            my_folder_dir / name
            for name in [
                'a_0001.jpg',
                'a_0002.jpg',
                'a_0003.jpg',
                'a_0004.jpg',
            ]
        ]
        actual = datatools.list_files(my_folder_dir, '.jpg')
        assert expected == actual

    def test_not_lexicographic(self, tmp_path):
        my_folder_dir = tmp_path / 'my_folder'
        fns = [
            my_folder_dir / name
            for name in [
                'a_0002.jpg',
                'a_0004.jpg',
                'a_0001.jpg',
                'a_0003.jpg',
            ]
        ]
        my_folder_dir.mkdir()
        for fn in fns:
            fn.touch()

        expected = [
            my_folder_dir / name
            for name in [
                'a_0002.jpg',
                'a_0004.jpg',
                'a_0001.jpg',
                'a_0003.jpg',
            ]
        ]
        actual = datatools.list_files(
            my_folder_dir, '.jpg', sort_ascending=False
        )
        assert datatools.lists_have_same_elements(expected, actual)


class TestListSubdirFiles:
    def test_naive(self, tmp_path):
        original_frames_dir = tmp_path / 'original_frames'
        video1_dir = original_frames_dir / 'video1'
        video2_dir = original_frames_dir / 'video2'
        video1_0001_fn = video1_dir / 'video1_0001.jpg'
        video1_0002_fn = video1_dir / 'video1_0002.jpg'
        video2_0001_fn = video1_dir / 'video2_0001.jpg'
        video2_0002_fn = video1_dir / 'video2_0002.jpg'

        original_frames_dir.mkdir()
        video1_dir.mkdir()
        video2_dir.mkdir()
        video1_0001_fn.touch()
        video1_0002_fn.touch()
        video2_0001_fn.touch()
        video2_0002_fn.touch()
        expected = [
            video1_0001_fn,
            video1_0002_fn,
            video2_0001_fn,
            video2_0002_fn,
        ]
        actual = datatools.list_subdir_files(original_frames_dir, '.jpg')
        assert datatools.lists_have_same_elements(expected, actual)
