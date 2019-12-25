import torch
import torch.nn as nn
import torch.nn.functional as F
import math





'''
Script provides functional interface for Mish activation function.
Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
https://arxiv.org/abs/1908.08681v1
'''
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))



class BetaMish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta=1.5
        return x * torch.tanh(torch.log(torch.pow((1+torch.exp(x)),beta)))


'''
Swish - https://arxiv.org/pdf/1710.05941v1.pdf
'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.



class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, act, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            act
        )
        self.fc = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.fc(y)
        return torch.mul(x, y)






NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'PReLU': nn.PReLU(),
    'ReLu6': nn.ReLU6(inplace=True),
    'Mish': Mish(),
    'BetaMish': BetaMish(),
    'Swish': Swish(),
    'Hswish': Hswish(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
}