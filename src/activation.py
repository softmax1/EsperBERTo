# Downloaded on 2023-08-23 from https://www.lyndonduong.com/saving-activations/

from collections import defaultdict
from functools import partial
from typing import DefaultDict, Tuple, List, Optional

from torch import Tensor, mean, pow
from torch.nn import Module


def kurtosis(out: Tensor) -> float:
    diffs = out - mean(out)
    stdev = pow(mean(pow(diffs, 2.)), 0.5)
    excess_kurtosis = mean(pow(diffs / stdev, 4.)) - 3.
    return excess_kurtosis.item()


def save_activations_kurtosis(
        activations: DefaultDict,
        name: str,
        module: Module,
        inp: Tuple,
        out: Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    activations[name].append(kurtosis(out.detach()))


def register_activation_hooks(
        model: Module,
        layers_to_save: Optional[List[str]] = None
) -> DefaultDict[str, List[float]]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save. If None, save all layers

    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = defaultdict(list)

    for name, module in model.named_modules():
        if layers_to_save is None or name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations_kurtosis, activations_dict, name)
            )
    return activations_dict
