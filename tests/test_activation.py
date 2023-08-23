from pytest import mark
from torch import randn
from torch.nn import Module, Conv2d
from torch.nn.functional import relu

from src.activation import register_activation_hooks


class Net(Module):
    """Simple two layer conv net"""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        y = relu(self.conv1(x))
        z = relu(self.conv2(y))
        return z


@mark.parametrize("acts_to_save", [None, "conv1,conv2"])
def test_register_activation_hooks(acts_to_save):
    mdl = Net()
    to_save = None if acts_to_save is None else acts_to_save.split(',')

    # register fwd hooks in specified layers
    saved_activations = register_activation_hooks(mdl, layers_to_save=to_save)

    # run twice, then assert each created lists for conv1 and conv2, each with length 2
    num_fwd = 2
    images = [randn(10, 3, 256, 256) for _ in range(num_fwd)]
    for _ in range(num_fwd):
        mdl(images[_])

    for activation in saved_activations:
        assert len(saved_activations[activation]) == num_fwd
