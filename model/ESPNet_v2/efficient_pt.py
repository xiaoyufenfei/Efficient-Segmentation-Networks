#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

from torch import nn
import math
from model.ESPNet_v2.cnn_utils import CBR

class EfficientPWConv(nn.Module):
    def __init__(self, nin, nout):
        super(EfficientPWConv, self).__init__()
        self.wt_layer = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                        nn.Sigmoid()
                    )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)

        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)