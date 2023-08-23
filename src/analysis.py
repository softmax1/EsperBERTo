from collections import defaultdict
from functools import partial
from json import dump
from pathlib import Path
from typing import List, Dict, Any, DefaultDict, Tuple, Optional

from torch import Tensor, mean, pow
from torch.nn import Module
from transformers import PreTrainedModel


# Statistics
def kurtosis(out: Tensor) -> float:
    diffs = out - mean(out)
    stdev = pow(mean(pow(diffs, 2.)), 0.5)
    excess_kurtosis = mean(pow(diffs / stdev, 4.)) - 3.
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


# Analysis
def compute_activation_statistics(saved_activation_kurtosis: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    activation_stats = dict()
    for activation in saved_activation_kurtosis:
        activation_stats[activation] = compute_avg_and_std(saved_activation_kurtosis[activation])
    return activation_stats


def compute_weight_statistics(model: PreTrainedModel) -> Dict[str, Dict[str, float]]:
    layer_kurtosis = dict()
    for name, param in model.named_parameters():
        layer_kurtosis[name] = kurtosis(param)

    weight_stats = defaultdict(list)
    for name in layer_kurtosis:
        if 'attention.output' in name:
            weight_stats[name].append(layer_kurtosis[name])

    return {name: compute_avg_and_std(kurtoses) for name, kurtoses in weight_stats.items()}
