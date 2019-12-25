from collections import OrderedDict
import os
import numpy as np


def convert_state_dict(state_dict):
    """
    Converts a state dict saved from a dataParallel module to normal module state_dict inplace
    Args:   
        state_dict is the loaded DataParallel model_state
    """
    state_dict_new = OrderedDict()
    # print(type(state_dict))
    for k, v in state_dict.items():
        # print(k)
        name = k[7:]  # remove the prefix module.
        # My heart is borken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
    return state_dict_new
