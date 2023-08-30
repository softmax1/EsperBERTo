from collections import defaultdict
from typing import List, Dict, Any

from torch.nn import Module
from flash_attention_softmax_n.analysis import compute_weight_statistics


def compute_avg_and_std(array: List[float]) -> Dict[str, float]:
    avg = sum(array) / len(array)
    second_moment = sum([x**2 for x in array]) / len(array)
    std = (second_moment - avg**2)**0.5
    return {'avg': avg, 'std': std}


def process_activation_stats(saved_activation_stats: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    activation_stats = defaultdict(list)
    for name in saved_activation_stats:
        if 'attention.output' in name:
            x = -2 if name.endswith('attention.output') else -3
            truncated_name = '.'.join(name.split('.')[x:])
            activation_stats[truncated_name].append(saved_activation_stats[name]['kurtosis'])

    activation_avg_attn_kurt = {name: compute_avg_and_std(kurtoses) for name, kurtoses in activation_stats.items()}
    return {'activation_stats': saved_activation_stats, 'activation_avg_attn_kurt': activation_avg_attn_kurt}


def process_weight_stats(model: Module) -> Dict[str, Dict[str, float]]:
    saved_weight_stats = compute_weight_statistics(model)

    weight_stats = defaultdict(list)
    for name in saved_weight_stats:
        if 'attention.output' in name:
            truncated_name = '.'.join(name.split('.')[-4:])
            weight_stats[truncated_name].append(saved_weight_stats[name]['kurtosis'])

    weight_avg_attn_kurt = {name: compute_avg_and_std(kurtoses) for name, kurtoses in weight_stats.items()}
    return {'weight_stats': saved_weight_stats, 'weight_avg_attn_kurt': weight_avg_attn_kurt}
