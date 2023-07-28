from torch import Tensor, index_select, arange
from torch.cuda import is_available
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
    # select the indices to keep
    # note that because we're creating this tensor it won't automatically be placed on the correct device
    device = 'cuda' if is_available() else 'cpu'
    index = arange(input.size()[-1], device=device)
    # un-pad the result
    return index_select(padded_output, dim=-1, index=index)


def softmax_1(input: Tensor) -> Tensor:
    return softmax_n(input, 1)
