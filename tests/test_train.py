from binarization import future_training


def test_get_starting_epoch_id_when_resuming():
    ckpt_path_to_resume = 'hola_5.pth'
    expected = 6
    actual = future_training.get_starting_epoch_id(ckpt_path_to_resume)
    assert expected == actual


def test_get_starting_epoch_id_when_resuming_without_an_epoch_id():
    ckpt_path_to_resume = 'hola.pth'
    expected = 0
    actual = future_training.get_starting_epoch_id(ckpt_path_to_resume)
    assert expected == actual


def test_get_starting_epoch_id_when_not_resuming():
    ckpt_path_to_resume = None
    expected = 0
    actual = future_training.get_starting_epoch_id(ckpt_path_to_resume)
    assert expected == actual
