from torch import Tensor, exp, multiply, add, divide, index_select, IntTensor
from torch.nn import ConstantPad1d
from torch.nn.functional import softmax


def softmax_n(input: Tensor, n: int) -> Tensor:
    """
    softmax_n(z_i) = exp(z_i) / (n + sum_j exp(z_j))
    """
    # instaniate a padder
    zero_padding = ConstantPad1d((0, n), 0)
    # pad the input with n 0s along the last dimension
    padded_input = zero_padding(input)
    # compute the softmax with the padded input
    padded_output = softmax(padded_input, dim=-1)
    # un-pad the result
    return index_select(padded_output, -1, IntTensor(range(input.size()[-1])))


def softmax_1(input: Tensor) -> Tensor:
    return softmax_n(input, 1)
