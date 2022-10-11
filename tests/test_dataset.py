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
