#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from torch import nn
from model.ESPNet_v2.eesp import DownSampler, EESP
from model.ESPNet_v2.espnet_utils import CBR
from torch.nn import init
import torch.nn.functional as F
from model.ESPNet_v2 import espnetv2_config as config_all
from model.ESPNet_v2.print_utils import *

class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, args):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()

        # ====================
        # Network configuraiton
        # ====================
        try:
            num_classes = args.num_classes
        except:
            # if not specified, default to 1000 for imageNet
            num_classes = 1000  # 1000 for imagenet

        try:
            channels_in = args.channels
        except:
            # if not specified, default to RGB (3)
            channels_in = 3

        s = args.s
        if not s in config_all.sc_ch_dict.keys():
            print_error_message('Model at scale s={} is not suppoerted yet'.format(s))
            exit(-1)

        out_channel_map = config_all.sc_ch_dict[args.s]
        reps_at_each_level = config_all.rep_layers

        recept_limit = config_all.recept_limit  # receptive field at each spatial level
        K = [config_all.branches]*len(recept_limit) # No. of parallel branches at different level

        # True for the shortcut connection with input
        self.input_reinforcement = config_all.input_reinforcement

        assert len(K) == len(recept_limit), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(out_channel_map[0], out_channel_map[1], k=K[0], r_lim=recept_limit[0], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(out_channel_map[1], out_channel_map[2], k=K[1], r_lim=recept_limit[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps_at_each_level[1]):
            self.level3.append(EESP(out_channel_map[2], out_channel_map[2], stride=1, k=K[2], r_lim=recept_limit[2]))

        self.level4_0 = DownSampler(out_channel_map[2], out_channel_map[3], k=K[2], r_lim=recept_limit[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps_at_each_level[2]):
            self.level4.append(EESP(out_channel_map[3], out_channel_map[3], stride=1, k=K[3], r_lim=recept_limit[3]))

        self.level5_0 = DownSampler(out_channel_map[3], out_channel_map[4], k=K[3], r_lim=recept_limit[3]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps_at_each_level[3]):
            self.level5.append(EESP(out_channel_map[4], out_channel_map[4], stride=1, k=K[4], r_lim=recept_limit[4]))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        self.level5.append(CBR(out_channel_map[4], out_channel_map[4], 3, 1, groups=out_channel_map[4]))
        self.level5.append(CBR(out_channel_map[4], out_channel_map[5], 1, 1, groups=K[4]))

        self.classifier = nn.Linear(out_channel_map[5], num_classes)
        self.config = out_channel_map
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)


if __name__ == '__main__':
    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    for scale in config_all.sc_ch_dict.keys():
        for size in [224]:
            # args.num_classes = 1000
            imSz = size
            args.s = scale
            args.channels = 3

            model = EESPNet(args)
            input = torch.randn(1, 3, size, size)
            print_info_message('Scale: {}, ImSize: {}x{}'.format(scale, size, size))
            print_info_message('Flops: {:.2f} million'.format(compute_flops(model, input)))
            print_info_message('Params: {:.2f} million'.format(model_parameters(model)))
            print('\n')