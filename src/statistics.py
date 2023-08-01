from json import dump
from pathlib import Path
from typing import Dict, Any, List

from torch import mean, pow
from transformers import PreTrainedModel


def compute_layer_statistics(model: PreTrainedModel) -> Dict[str, Dict[str, float]]:
    results = dict()
    for name, param in model.named_parameters():
        mean_val = param.data.mean()
        diffs = param.data - mean_val
        var = mean(pow(diffs, 2.0))
        std = pow(var, 0.5)
        zscores = diffs / std
        skews = mean(pow(zscores, 3.0))
        kurtosis = mean(pow(zscores, 4.0)) - 3.0

        results[name] = {
            'mean': mean_val.item(),
            'var': var.item(),
            'std': std.item(),
            'skews': skews.item(),
            'kurtosis': kurtosis.item()
        }
    return results


def compute_avg_and_std(array: List[float]) -> Dict[str, float]:
    avg = sum(array) / len(array)
    second_moment = sum([x**2 for x in array]) / len(array)
    std = (second_moment - avg**2)**0.5
    return {'avg': avg, 'std': std}


def compute_average_statistics(layer_statistics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    dense_weight_kurtoses = list()
    dense_bias_kurtoses = list()
    layernorm_weight_kurtoses = list()
    layernorm_bias_kurtoses = list()

    for name in layer_statistics:
        if 'attention.output' in name:
            if name.endswith('dense.weight'):
                dense_weight_kurtoses.append(layer_statistics[name]['kurtosis'])
            if name.endswith('dense.bias'):
                dense_bias_kurtoses.append(layer_statistics[name]['kurtosis'])
            if name.endswith('LayerNorm.weight'):
                layernorm_weight_kurtoses.append(layer_statistics[name]['kurtosis'])
            if name.endswith('LayerNorm.bias'):
                layernorm_bias_kurtoses.append(layer_statistics[name]['kurtosis'])

    return {
        'layer_statistics': layer_statistics,
        'average_kurtosis': {
            'attention.output.dense.weight': compute_avg_and_std(dense_weight_kurtoses),
            'attention.output.dense.bias': compute_avg_and_std(dense_bias_kurtoses),
            'attention.output.LayerNorm.weight': compute_avg_and_std(layernorm_weight_kurtoses),
            'attention.output.LayerNorm.bias': compute_avg_and_std(layernorm_bias_kurtoses)
        }
    }


def compute_statistics(model: PreTrainedModel) -> Dict[str, Dict[str, Any]]:
    layer_statistics = compute_layer_statistics(model)
    return compute_average_statistics(layer_statistics)


def save_statistics(results: Dict[str, Dict[str, Any]], output_dir: str):
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    model_name = output_dir.split("/")[1]
    filepath = results_dir / f"{model_name}.json"

    with filepath.open(mode='w') as fp:
        dump(results, fp, indent=4)
