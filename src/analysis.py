from collections import defaultdict
from functools import partial
from json import dump
from pathlib import Path
from typing import List, Dict, Any, DefaultDict, Tuple, Optional

from torch import Tensor, mean, pow
from torch.nn import Module
from transformers import PreTrainedModel


# Statistics
def kurtosis(x: Tensor) -> float:
    """
    kurtosis[x] = E[y^2] / (E[y])^2, where y = (x - E[x])^2
    excess_kurtosis = kurtosis - 3
    """
    y = pow(x - mean(x), 2)
    excess_kurtosis = mean(pow(y, 2)) / pow(mean(y), 2) - 3.
    return excess_kurtosis.item()


def compute_avg_and_std(array: List[float]) -> Dict[str, float]:
    avg = sum(array) / len(array)
    second_moment = sum([x**2 for x in array]) / len(array)
    std = (second_moment - avg**2)**0.5
    return {'avg': avg, 'std': std}


# I/O
def save_results(results: Dict[str, Dict[str, Any]], output_dir: str):
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    model_name = output_dir.split("/")[1]
    filepath = results_dir / f"{model_name}.json"

    with filepath.open(mode='w') as fp:
        dump(results, fp, indent=4)


# Activation Kurtosis Hook
def save_activations_kurtosis(
        activations: DefaultDict,
        name: str,
        module: Module,
        inp: Tuple,
        out: Tensor
) -> None:
    """
    PyTorch Forward hook to compute moving average of kurtosis  at each forward pass.
    Mutates specified dict objects with each fwd pass.
    """
    k = kurtosis(out[0].detach())
    n, mu = activations[name]
    activations[name] = [n + 1, (n * mu + k) / (n + 1)]


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
    activations_dict = defaultdict(lambda: [0, 0.])

    for name, module in model.named_modules():
        if layers_to_save is None or name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations_kurtosis, activations_dict, name)
            )
    return activations_dict


# Analysis
def compute_activation_statistics(saved_activation_kurtosis: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    activation_kurtosis = {activation: val[1] for activation, val in saved_activation_kurtosis.items()}

    activation_stats = defaultdict(list)
    for name in activation_kurtosis:
        if 'attention.output' in name:
            x = -2 if name.endswith('attention.output') else -3
            truncated_name = '.'.join(name.split('.')[x:])
            activation_stats[truncated_name].append(activation_kurtosis[name])

    activation_avg_attn_kurt = {name: compute_avg_and_std(kurtoses) for name, kurtoses in activation_stats.items()}
    return {'activation_kurtosis': activation_kurtosis, 'activation_avg_attn_kurt': activation_avg_attn_kurt}


def compute_weight_statistics(model: PreTrainedModel) -> Dict[str, Dict[str, float]]:
    weight_kurtosis = {name: kurtosis(param) for name, param in model.named_parameters()}

    weight_stats = defaultdict(list)
    for name in weight_kurtosis:
        if 'attention.output' in name:
            truncated_name = '.'.join(name.split('.')[-4:])
            weight_stats[truncated_name].append(weight_kurtosis[name])

    weight_avg_attn_kurt = {name: compute_avg_and_std(kurtoses) for name, kurtoses in weight_stats.items()}
    return {'weight_kurtosis': weight_kurtosis, 'weight_avg_attn_kurt': weight_avg_attn_kurt}
