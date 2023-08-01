from pytest import fixture, approx

from torch import Tensor, log
from torch.nn.functional import softmax

from src.functional import softmax_n_shifted_zeros, softmax_1, softmax_n_with_padding


@fixture(scope='session')
def expected_numerators():
    return [
        [1, 3, 6],
        [3, 1, 4],
        [1 / 6, 1 / 3, 1 / 2],
        [0.5, 1.5, 3],
        [100, 200, 300],
        [1 / 600, 1 / 300, 1 / 200],
        [2 / 7, 4 / 7, 8 / 7]
    ]


@fixture(scope='session')
def expected_denominators(expected_numerators):
    return [sum(test_case) for test_case in expected_numerators]


@fixture(scope='session')
def input_data(expected_numerators):
    return log(Tensor(expected_numerators))


def test_softmax_n_with_padding(input_data, expected_numerators, expected_denominators):
    idx_max, jdx_max = input_data.size()

    output_0_builtin = softmax(input_data, dim=-1)
    assert output_0_builtin.size() == input_data.size()
    for idx in range(idx_max):
        for jdx in range(jdx_max):
            expected_answer = expected_numerators[idx][jdx] / expected_denominators[idx]
            assert output_0_builtin[idx][jdx].item() == approx(expected_answer)

    output_0 = softmax_n_with_padding(input_data, 0, dim=-1)
    assert output_0.size() == input_data.size()
    for idx in range(idx_max):
        for jdx in range(jdx_max):
            assert output_0[idx][jdx].item() == approx(output_0_builtin[idx][jdx])

    for n in range(1, 7, 3):
        output = softmax_n_with_padding(input_data, n, dim=-1)
        assert output.size() == input_data.size()
        for idx in range(idx_max):
            for jdx in range(jdx_max):
                expected_answer = expected_numerators[idx][jdx] / (expected_denominators[idx] + n)
                assert output[idx][jdx].item() == approx(expected_answer)


def test_softmax_n_shifted_zeros(input_data, expected_numerators, expected_denominators):
    idx_max, jdx_max = input_data.size()

    output_0_builtin = softmax(input_data, dim=-1)

    output_0 = softmax_n_shifted_zeros(input_data, 0, dim=-1)
    assert output_0.size() == input_data.size()
    for idx in range(idx_max):
        for jdx in range(jdx_max):
            assert output_0[idx][jdx].item() == approx(output_0_builtin[idx][jdx])

    for n in range(1, 7, 3):
        output = softmax_n_shifted_zeros(input_data, n, dim=-1)
        assert output.size() == input_data.size()
        for idx in range(idx_max):
            for jdx in range(jdx_max):
                expected_answer = expected_numerators[idx][jdx] / (expected_denominators[idx] + n)
                assert output[idx][jdx].item() == approx(expected_answer)


def test_softmax_1(input_data, expected_numerators, expected_denominators):
    idx_max, jdx_max = input_data.size()

    output_1 = softmax_1(input_data, dim=-1)
    assert output_1.size() == input_data.size()
    for idx in range(idx_max):
        for jdx in range(jdx_max):
            expected_answer = expected_numerators[idx][jdx] / (expected_denominators[idx] + 1)
            assert output_1[idx][jdx].item() == approx(expected_answer)
