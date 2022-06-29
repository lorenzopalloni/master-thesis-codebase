import pytest  # noqa E401
import torch

from binarization import models


class TestDiscriminator:
    def test__out_channels_helper_i_0(self):

        i = 0
        num_channels = 32  # -> initilial value, when i == 0
        all_out_channels = [
            3,  # input_num_channels
        ]  # all_out_channels[-1] -> default value, when i > 0 and i % 2 == 0
        dis = models.Discriminator()

        actual = dis._out_channels_helper(
            i=i,
            default=all_out_channels[-1],
            init=num_channels
        )
        expected = num_channels

        assert expected == actual

    def test__out_channels_helper_i_1(self):

        i = 1
        num_channels = 32  # -> initilial value, when i == 0
        all_out_channels = [
            3,  # input_num_channels
            32,
        ]  # all_out_channels[-1] -> default value, when i > 0 and i % 2 == 0
        dis = models.Discriminator()

        actual = dis._out_channels_helper(
            i=i,
            default=all_out_channels[-1],
            init=num_channels
        )
        expected = all_out_channels[-1]

        assert expected == actual

    def test__out_channels_helper_i_even(self):

        i = 2
        num_channels = 32  # -> initilial value, when i == 0
        all_out_channels = [
            3,  # input_num_channels
            32,
            32,
        ]  # all_out_channels[-1] -> default value, when i > 0 and i % 2 == 0
        dis = models.Discriminator()

        actual = dis._out_channels_helper(
            i=i,
            default=all_out_channels[-1],
            init=num_channels
        )
        expected = all_out_channels[-1] * 2

        assert expected == actual

    def test__out_channels_helper_i_odd(self):

        i = 3
        num_channels = 32  # -> initilial value, when i == 0
        all_out_channels = [
            3,  # input_num_channels
            32,
            32,
            64,
        ]  # all_out_channels[-1] -> default
        dis = models.Discriminator()

        actual = dis._out_channels_helper(
            i=i,
            default=all_out_channels[-1],
            init=num_channels
        )
        expected = all_out_channels[-1]

        assert expected == actual


class TestConvBlock:
    def test__activation_layer_helper_none(self):
        actual = models.ConvBlock._activation_layer_helper(None)
        expected = None
        assert actual == expected

    def test__activation_layer_helper_prelu(self):
        actual = models.ConvBlock._activation_layer_helper('prelu')
        expected = torch.nn.PReLU()
        assert actual.__class__ == expected.__class__

    def test__activation_layer_helper_leakyrelu(self):
        actual = models.ConvBlock._activation_layer_helper('leakyrelu')
        expected = torch.nn.LeakyReLU(0.2)
        assert actual.__class__ == expected.__class__

    def test__activation_layer_helper_tanh(self):
        actual = models.ConvBlock._activation_layer_helper('tanh')
        expected = torch.nn.Tanh()
        assert actual.__class__ == expected.__class__

    def test__activation_layer_helper_some_upper_cases(self):
        actual = models.ConvBlock._activation_layer_helper('PreLU')
        expected = torch.nn.PReLU()
        assert actual.__class__ == expected.__class__

    def test__activation_layer_helper_mispelled(self):
        with pytest.raises(AssertionError):
            models.ConvBlock._activation_layer_helper('PreL')
