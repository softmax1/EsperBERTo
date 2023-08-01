from pytest import fixture, approx
from torch import rand
from torch.nn.functional import softmax

from src.functional import softmax_1, clipped_softmax


@fixture(scope='session')
def input_data():
    """
    Uniform [-0.03, 0.03]

    The inputs to the softmax in self-attention are not logits even though the output of a softmax has an interpretation as a probability.
    Instead, the input measures whether token i attends to token j.
    If token i does not attend to token j, then q_i dot k_j should be close to zero.
    In the case that an attention has nothing relevant to add, then all the inputs to the (modified) softmax should be small in magnitude.
    """
    return (rand(100, 100) - 0.5) * 3 / 50


def test_clipped_softmax(input_data):
    """
    The clipped softmax eliminates the noise.
    """
    output = clipped_softmax(input_data, 1.03, -0.03, dim=-1)
    assert output.max().item() == 0.


def test_softmax_1(input_data):
    """
    The noise is still present when using softmax_n.
    The results for softmax_1 are quite similar to those of softmax_0.
    You can increase n, but then meaningful relationship between queries and keys will be impacted.
    """
    output_0 = softmax(input_data, dim=-1)
    output_1 = softmax_1(input_data, dim=-1)
    assert 0. < output_0.max().item() < 0.03
    assert 0. < output_1.max().item() < 0.03
    assert output_1 == approx(output_0, rel=0.01)
