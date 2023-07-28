from json import dump
from pathlib import Path
from typing import Dict, Any

from torch import mean, pow
from transformers import PreTrainedModel


def compute_statistics(model: PreTrainedModel) -> Dict[str, Dict[str, Any]]:
    results = dict()
    for name, param in model.named_parameters():
        name_results = dict()

        mean_val = param.data.mean()
        diffs = param.data - mean_val
        var = mean(pow(diffs, 2.0))
        std = pow(var, 0.5)
        zscores = diffs / std
        skews = mean(pow(zscores, 3.0))
        kurtosis = mean(pow(zscores, 4.0)) - 3.0

        name_results['mean'] = mean_val.item()
        name_results['var'] = var.item()
        name_results['std'] = std.item()
        name_results['skews'] = skews.item()
        name_results['kurtosis'] = kurtosis.item()

        results[name] = name_results
    return results


def save_statistics(results: Dict[str, Dict[str, Any]], output_dir: str):
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    model_name = output_dir.split("/")[1]
    filepath = results_dir / f"{model_name}.json"

    with filepath.open(mode='w') as fp:
        dump(results, fp, indent=4)
