from torch import Tensor, exp, multiply, add, divide
from torch.nn.functional import softmax


def softmax_n(input: Tensor, n: float) -> Tensor:
    """
    softmax(z_i) = exp(z_i) / sum_j exp(z_j)
    SM := E / Z

    softmax_n(z_i) = exp(z_i) / (n + sum_j exp(z_j))
    SMn := E / (n + Z)
    = E / (n + E / SM)
    = SM * E / (SM * n + E)

    """
    if n == 0:
        return softmax(input, dim=-1)
    else:
        sm = softmax(input, dim=-1)
        e = exp(input)
        return divide(multiply(sm, e), add(multiply(sm, n), e))


def softmax_1(input: Tensor) -> Tensor:
    return softmax_n(input, 1)
