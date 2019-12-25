
import math
import torch
from torch.nn import init
import torch.nn.functional as F
from model.ESPNet_v2.cnn_utils import *


__author__ = "Sachin Mehta"
__version__ = "1.0.1"
__maintainer__ = "Sachin Mehta"



class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'): #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param g: number of groups to be used in the feature map reduction step.
        '''
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        #assert nIn%k == 0, "Number of input channels ({}) should be divisible by # of branches ({})".format(nIn, k)
        #assert n % k == 0, "Number of output channels ({}) should be divisible by # of branches ({})".format(n, k)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        #self.bn = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
            #self.bn.append(nn.BatchNorm2d(n))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            # We donot combine the branches that have the same effective receptive (3x3 in our case)
            # because there are no holes in those kernels.
            out_k = out_k + output[k - 1]
            #apply batch norm after fusion and then append to the list
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # Aggregate the feature maps using point-wise convolution
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # as Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has two parallel branches: (1) avg pooling
    and (2) EESP block with stride of 2. The output feature maps of these branches
    are then concatenated and thresholded using an activation function (PReLU in our
    case) to produce the final output.
    '''

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param g: number of groups to be used in the feature map reduction step.
        '''
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(config_inp_reinf, config_inp_reinf, 3, 1),
                CB(config_inp_reinf, nout, 1, 1)
            )
        self.act =  nn.PReLU(nout)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)
        if input2 is not None:
            #assuming the input is a square image
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output) #self.act(output)

class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, classes=19, s=1):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        reps = [0, 3, 7, 3]  # how many times EESP blocks should be repeated.
        channels = 3

        r_lim = [13, 11, 9, 7, 5]  # receptive field at each spatial level
        K = [4]*len(r_lim) # No. of parallel branches at different levels

        base = 32 #base configuration
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i== 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            ValueError('Configuration not supported')

        #print('Config: ', config)

        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels, config[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=r_lim[0], reinf=self.input_reinforcement)  # out = 56
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=r_lim[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2], r_lim=r_lim[2]))

        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=r_lim[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3], r_lim=r_lim[3]))

        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=r_lim[3]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4], r_lim=r_lim[4]))

        # expand the feature maps using depth-wise separable convolution
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))



        #self.level5_exp = nn.ModuleList()
        #assert config[5]%config[4] == 0, '{} should be divisible by {}'.format(config[5], config[4])
        #gr = int(config[5]/config[4])
        #for i in range(gr):
        #    self.level5_exp.append(CBR(config[4], config[4], 1, 1, groups=pow(2, i)))

        self.classifier = nn.Linear(config[5], classes)
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

    def forward(self, input, p=0.2, seg=True):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # out_l2_inp_rein
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sampled
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        if not seg:
            out_l5_0 = self.level5_0(out_l4)  # down-sampled
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)

            #out_e = []
            #for layer in self.level5_exp:
            #    out_e.append(layer(out_l5))
            #out_exp = torch.cat(out_e, dim=1)



            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=p, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)

            return self.classifier(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


