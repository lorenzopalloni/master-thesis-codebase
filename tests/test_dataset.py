# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
from binarization.dataset import BufferGenerator, get_train_val_test_indexes


class TestBufferGenerator:
    def test_init(self):
        init_list = list('abecedario')
        buffer_generator = BufferGenerator(init_list, buffer_size=2, shuffle=False)
        actual = []
        for buffer in buffer_generator:
            actual.append(buffer)
        expected = [['a', 'b'], ['e', 'c'], ['e', 'd'], ['a', 'r'], ['i', 'o']]
        assert all(x == y for x, y in zip(expected, actual))

    def test_random_init(self):
        init_list = list('abecedario')
        buffer_generator = BufferGenerator(init_list, buffer_size=3, shuffle=True)
        actual_values = []
        for buffer in buffer_generator:
            for elem in buffer:
                actual_values.append(elem)
            assert all(x in init_list for x in buffer)
        assert len(actual_values) == len(init_list) - 1


def test_get_train_val_test_indexes():
    n_examples = 100
    train_indexes, val_indexes, test_indexes = (
        get_train_val_test_indexes(
            n=n_examples,
            val_ratio=0.33,
            test_ratio=0.05,
        )
    )
    assert (
        len(train_indexes),
        len(val_indexes),
        len(test_indexes)
    ) == (62, 33, 5)
    union_indexes = set(train_indexes).union(
        set(val_indexes)
    ).union(set(test_indexes))
    assert union_indexes == set(range(n_examples))
    assert len(union_indexes) == len(set(range(n_examples)))
