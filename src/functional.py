from torch import Tensor, index_select, arange, exp, divide, subtract, multiply, add
from torch.cuda import is_available
from torch.nn import ConstantPad1d
from torch.nn.functional import softmax


def softmax_n_shifted_zeros(input: Tensor, n: int) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=-1, keepdim=True).values
    # shift the input to prevent overflow (and underflow in the denominator)
    shifted_inputs = subtract(input, input_maxes)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = exp(shifted_inputs)
    original_denominator = numerator.sum(dim=-1, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_zeros = multiply(input_maxes, -1)
    # and then add this contribution to the denominator
    denominator = add(original_denominator, multiply(exp(shifted_zeros), n))
    return divide(numerator, denominator)


def softmax_1(input: Tensor) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (1 + \sum_j exp(x_j))$

    After a small amount of testing, the "shifted zeros" approach appears to be faster.
    I am definitely open to suggestions on which approach is better though.
    """
    return softmax_n_shifted_zeros(input, 1)


def softmax_n_with_padding(input: Tensor, n: int) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + sum_j exp(x_j))$

    The idea here is to pad the input with zeros.
    That way the softmax, with its stable implementation under the hood, can naturally be used.
    Afterwards we need to un-pad the output.
    """
    # instaniate a padder
    zero_padding = ConstantPad1d((0, n), 0)
    # pad the input with n 0s along the last dimension
    padded_input = zero_padding(input)
    # compute the softmax with the padded input
    padded_output = softmax(padded_input, dim=-1)
    # select the indices to keep
    # note that because we're creating this tensor it won't automatically be placed on the correct device
    device = 'cuda' if is_available() else 'cpu'
    indices_to_keep = arange(input.size()[-1], device=device)
    # un-pad the result
    return index_select(padded_output, dim=-1, index=indices_to_keep)
