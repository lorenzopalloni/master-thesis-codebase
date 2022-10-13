from binarization import dataset


class TestBuffer:
    def test_init(self):
        init_list = [
            'mwqosb',
            'kzbhhuyfwrp',
            'jjpphrsc',
            'xpumove',
            'mcpbbuy',
            'wrjhmmg',
            'uwxye',
            'krijbstophej',
            'bxpvtmobuhl',
            'ehtlaf'
        ]

        iterator = iter(dataset.Buffer(init_list, buffer_size=3, shuffle=True))
        actual_values = []
        for batch in iterator:
            for elem in batch:
                actual_values.append(elem)
            assert all(x in init_list for x in batch)
            assert len(set(batch)) == len(batch)
        assert len(set(actual_values)) == len(set(init_list)) - 1
    

def test_get_train_val_test_indexes():
    n = 100
    train_indexes, val_indexes, test_indexes = (
        dataset.get_train_val_test_indexes(
            n=n,
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
    assert union_indexes == set(range(n))
    assert len(union_indexes) == len(set(range(n)))
