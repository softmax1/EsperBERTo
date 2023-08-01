from pytest import fixture
from torch import Tensor, isnan

from src.functional import softmax_n_with_padding, softmax_n_shifted_zeros, softmax_n_naive


@fixture(scope='session')
def input_data():
    """
    exp([12., 89., 710.]) will lead to overflow at half-, single-, or double-precision
    """
    return Tensor([12., 89., 710.])


def test_softmax_n_with_padding(input_data):
    output = softmax_n_with_padding(input_data, 1, dim=-1)
    assert output.sum() == 1


def test_softmax_n_shifted_zeros(input_data):
    output = softmax_n_shifted_zeros(input_data, 1, dim=-1)
    assert output.sum() == 1


def test_softmax_n_naive(input_data):
    output = softmax_n_naive(input_data, 1, dim=-1)
    assert isnan(output).sum() == 2  # the computation is done in single-precision
