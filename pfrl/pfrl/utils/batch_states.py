from typing import Any, Callable, Sequence

import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate, default_convert


def _to_recursive(batched: Any, device: torch.device) -> Any:
    if isinstance(batched, torch.Tensor):
        return batched.to(device)
    elif isinstance(batched, list):
        return [x.to(device) for x in batched]
    elif isinstance(batched, tuple):
        return tuple(x.to(device) for x in batched)
    else:
        raise TypeError("Unsupported type of data")


def batch_states(
    states: Sequence[Any], device: torch.device, phi: Callable[[Any], Any]
) -> Any:
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        device (module): CPU or GPU the data should be placed on
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    # 以下为代码调试：
    # 必须要在该程序中运行如下语句，因为环境产生的观测包含三个矩阵，
    # 而三个矩阵被合成了一个元组，必须将第一项拆分出来
    if isinstance(states[0], tuple):
        states = states[0]

    # count = 0
    # for s in states:
    #     print(phi(s))
    #     count += 1
    # print(count)

    # 提取特征并转换成tensor数据类型
    features = [phi(s) for s in states]
    features = [torch.as_tensor(b) for b in features]
    return _to_recursive(features, device)

    # 以下为作者编写的原始程序，但只需要将特征转换为tensor类型即可，不需要如下程序
    # return concat_examples(features, device=device)
    # collated_features = default_collate(features)  # 该语句为出问题的语句
    # if isinstance(features[0], tuple):
    #     collated_features = tuple(collated_features)
    # return _to_recursive(features, device)
