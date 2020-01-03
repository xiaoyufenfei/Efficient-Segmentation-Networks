#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn

# helper function for activations
def activation_fn(features, name='prelu', inplace=True):
    '''
    :param features: # of features (only for PReLU)
    :param name: activation name (prelu, relu, selu)
    :param inplace: Inplace operation or not
    :return:
    '''
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        NotImplementedError('Not implemented yet')
        exit()

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and activation function
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        :param act_name: Name of the activation function
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cbr(x)

class CB(nn.Module):
    '''
    This class implements convolution layer followed by batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cb = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=1),
            nn.BatchNorm2d(nOut),
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cb(x)


class BR(nn.Module):
    '''
    This class implements batch normalization and  activation function
    '''
    def __init__(self, nOut, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param act_name: Name of the activation function
        '''
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.br(x)


class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class DWConv(nn.Module):
    def __init__(self, nin):
        super(DWConv, self).__init__()
        self.dw_layer = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=3, stride=1, padding=1, bias=False, groups=nin),
            nn.BatchNorm2d(nin),
            nn.PReLU(nin)
        )

    def forward(self, x):
        return self.dw_layer(x)