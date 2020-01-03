import os
import torch
from model.ESPNet_v2.print_utils import print_info_message
import numpy as np

#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

'''
This file is mostly adapted from the PyTorch ImageNet example
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
Utility to save checkpoint or not
'''
def save_checkpoint(state, is_best, dir, extra_info='model', epoch=-1):
    check_pt_file = dir + os.sep + str(extra_info) + '_checkpoint.pth.tar'
    torch.save(state, check_pt_file)
    if is_best:
        #We only need best models weight and not check point states, etc.
        torch.save(state['state_dict'], dir + os.sep + str(extra_info) + '_best.pth')
    if epoch != -1:
        torch.save(state['state_dict'], dir + os.sep + str(extra_info) + '_ep_' + str(epoch) + '.pth')

    print_info_message('Checkpoint saved at: {}'.format(check_pt_file))


'''
Function to compute model parameters
'''
def model_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters()])/ 1e6

'''
function to compute flops
'''
def compute_flops(model, input=None):
    from model.ESPNet_v2.flops_compute import add_flops_counting_methods
    input = input if input is not None else torch.Tensor(1, 3, 224, 224)
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()

    _ = model(input)

    flops = model.compute_average_flops_cost()  # + (model.classifier.in_features * model.classifier.out_features)
    flops = flops / 1e6 / 2
    return flops