from pytest import fixture, approx

from torch import Tensor, log
from torch.nn.functional import softmax

from src.activation import softmax_n, softmax_1


@fixture(scope='session')
def expected_numerators():
    return [[1, 3, 6], [3, 1, 4]]


@fixture(scope='session')
def expected_denominators():
    return [10, 8]


@fixture(scope='session')
def input_data(expected_numerators):
    return log(Tensor(expected_numerators))


def test_softmax_n(input_data, expected_numerators, expected_denominators):
    output_0_builtin = softmax(input_data, dim=-1)
    assert output_0_builtin.size() == input_data.size()
    for idx in range(2):
        for jdx in range(3):
            assert output_0_builtin[idx][jdx].item() == approx(expected_numerators[idx][jdx] / expected_denominators[idx])

    output_0 = softmax_n(input_data, 0)
    assert output_0.size() == input_data.size()
    for idx in range(2):
        for jdx in range(3):
            assert output_0[idx][jdx].item() == approx(output_0_builtin[idx][jdx])

    output_1 = softmax_n(input_data, 1)
    assert output_1.size() == input_data.size()
    for idx in range(2):
        for jdx in range(3):
            assert output_1[idx][jdx].item() == approx(expected_numerators[idx][jdx] / (expected_denominators[idx] + 1))

    output_4 = softmax_n(input_data, 4)
    assert output_4.size() == input_data.size()
    for idx in range(2):
        for jdx in range(3):
            assert output_4[idx][jdx].item() == approx(expected_numerators[idx][jdx] / (expected_denominators[idx] + 4))


def test_softmax_1(input_data, expected_numerators, expected_denominators):
    output_1 = softmax_1(input_data)
    assert output_1.size() == input_data.size()
    for idx in range(2):
        for jdx in range(3):
            assert output_1[idx][jdx].item() == approx(expected_numerators[idx][jdx] / (expected_denominators[idx] + 1))
