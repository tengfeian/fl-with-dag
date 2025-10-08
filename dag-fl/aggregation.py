from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import copy
import math
from typing import List
from torch import nn
import torch
import numpy as np

def fedAvg(models):
    avg_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for key in avg_model.state_dict().keys():
            avg_model.state_dict()[key].copy_(
                sum(model.state_dict()[key] for model in models) / len(models)
            )
    return avg_model
#
# def fedAvg (model_list) -> nn.Module:
#     assert (len(model_list) > 1), "incorrect parameters"
#
#     m_last = model_list.pop()
#     agg_model = copy.deepcopy(m_last)
#
#     agg_m = agg_model.named_parameters()
#     agg_m_dict = dict(agg_m)
#
#     factor = 1/len(model_list)
#     # use m_top instead of agg_m for iteration, because agg_m.named_parameters() has already completed its iteration after the defination of agg_m_dict
#     for name, para in m_last.named_parameters():
#         avg = para.data * factor
#         for m in model_list:
#             m_dict = dict(m.named_parameters())
#             if name in m_dict:
#                 avg = avg + m_dict[name].data * factor
#         agg_m_dict[name].data.copy_(avg)
#     return agg_model

# def weighted_fedAvg(models_dict, datasize_dict) -> nn.Module:
#     assert (len(models_dict.keys()) > 1 and len(datasize_dict.keys()) >
#             1), "incorrect parameters"
#     assert (len(models_dict.keys()) == len(
#         datasize_dict.keys())), "incorrect parameters"
#
#     pop_id, m_top = models_dict.popitem()
#     agg_model = copy.deepcopy(m_top)
#
#     datasize_sum = 0
#     for size in datasize_dict.values():
#         datasize_sum += size
#
#     factors = {}
#     for id in datasize_dict.keys():
#         size = datasize_dict[id]
#         factors[id] = size/datasize_sum
#
#     agg_m = agg_model.named_parameters()
#     agg_m_dict = dict(agg_m)
#     # use m_top instead of agg_m for iteration, because agg_m.named_parameters() has already completed its iteration after the defination of agg_m_dict
#     for name, para in m_top.named_parameters():
#         weighted_avg = para.data * factors[pop_id]
#
#         for id in models_dict.keys():
#             m_dict = dict(models_dict[id].named_parameters())
#             if name in m_dict:
#                 weighted_avg = weighted_avg + m_dict[name].data * factors[id]
#         agg_m_dict[name].data.copy_(weighted_avg)
#
#     return agg_model