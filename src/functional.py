from typing import Optional

from torch import Tensor, index_select, arange, exp, divide, subtract, multiply, add, clip
from torch.cuda import is_available
from torch.nn.functional import softmax, pad

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int


def softmax_n(
        input: Tensor,
        n: Optional[float] = None,
        dim: Optional[int] = None,
        dtype: Optional[DType] = None
) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    if n is None:
        n = 0.
    shift = input.max(dim=dim, keepdim=True).values.detach()
    numerator = exp(input - shift)
    output = numerator / (n * exp(-shift) + numerator.sum(dim=dim, keepdim=True))
    return output if dtype is None else output.type(dtype=dtype)

# Everything below here is for demonstration purposes


def softmax_n_with_padding(input: Tensor, n: int, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + sum_j exp(x_j))$

    The idea here is to pad the input with zeros.
    That way the softmax, with its stable implementation under the hood, can naturally be used.
    Afterwards we need to un-pad the output.
    """
    # instaniate a padder
    if dim is None:
        raise NotImplementedError('The padding approach is currently only implemented for a specific dimension.')
    if dim >= 0:
        dim -= len(input.size())
    padding_size = -(2 * dim + 1) * (0,) + (n,)

    # pad the input with n 0s along the 'dim' dimension
    padded_input = pad(input, padding_size, value=0)
    # compute the softmax with the padded input
    padded_output = softmax(padded_input, dim=dim)
    # select the indices to keep
    # note that because we're creating this tensor it won't automatically be placed on the correct device
    indices_to_keep = arange(input.size()[dim], device=input.device)
    # un-pad the result
    output = index_select(padded_output, dim=dim, index=indices_to_keep)
    return output if dtype is None else output.type(dtype=dtype)


def softmax_1(input: Tensor, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (1 + \sum_j exp(x_j))$
    """
    return softmax_n_with_padding(input, 1, dim=dim, dtype=dtype)


def softmax_n_shifted_zeros(input: Tensor, n: int, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    # compute the maxes along the last dimension
    input_maxes = input.max(dim=dim, keepdim=True).values.detach()
    # shift the input to prevent overflow (and underflow in the denominator)
    shifted_inputs = subtract(input, input_maxes)
    # compute the numerator and softmax_0 denominator using the shifted input
    numerator = exp(shifted_inputs)
    original_denominator = numerator.sum(dim=dim, keepdim=True)
    # we need to shift the zeros in the same way we shifted the inputs
    shifted_zeros = multiply(input_maxes, -1)
    # and then add this contribution to the denominator
    denominator = add(original_denominator, multiply(exp(shifted_zeros), n))
    output = divide(numerator, denominator)
    return output if dtype is None else output.type(dtype=dtype)


def softmax_n_naive(input: Tensor, n: int, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    numerator = exp(input)
    output = numerator / (numerator.sum(dim=dim, keepdim=True) + n)
    return output if dtype is None else output.type(dtype=dtype)


def clipped_softmax(input: Tensor,
                    xi: float,
                    gamma: float,
                    dim: Optional[int] = None,
                    dtype: Optional[DType] = None
                    ) -> Tensor:
    assert xi >= 1.
    assert gamma <= 0.
    output = clip((xi - gamma) * softmax(input, dim=dim) + gamma, 0., 1.)
    return output if dtype is None else output.type(dtype=dtype)
