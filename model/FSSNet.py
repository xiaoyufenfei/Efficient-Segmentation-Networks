######################################################################################
#FSSNet: Fast Semantic Segmentation for Scene Perception
#Paper-Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8392426
######################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.activations import NON_LINEARITY


__all__ = ["FSSNet"]

# NON_LINEARITY = {
#     'ReLU': nn.ReLU(inplace=True),
#     'PReLU': nn.PReLU(),
# 	'ReLu6': nn.ReLU6(inplace=True)
# }

class InitialBlock(nn.Module):
    def __init__(self, ninput, noutput, non_linear='ReLU'):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput-ninput, eps=1e-3)
        self.relu = NON_LINEARITY[non_linear]

    def forward(self, input):
        output = self.relu(self.bn(self.conv(input)))
        output = torch.cat([output, self.pool(input)], 1)

        return output


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3,
                 padding=0, dropout_prob=0., bias=False, non_linear='ReLU'):
        super().__init__()
        # Store parameters that are needed later
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias),
        )

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2, no padding
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear]
        )
        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear]
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            NON_LINEARITY[non_linear]
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        # PReLU layer to apply after concatenating the branches
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x):
        # Main branch shortcut
        main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = self.out_prelu(main + ext)

        return out


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=2,
                 padding=0, dropout_prob=0., bias=False, non_linear='ReLU'):
        super().__init__()
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        # self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear]
        )
        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding,
                               output_padding=0, bias=bias),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear]
        )
        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            NON_LINEARITY[non_linear]
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x, x_pre):
        # Main branch shortcut         # here different origin paper, Fig 4 contradict to Fig 9
        main = x + x_pre

        main = self.main_conv1(main)     # 2. conv first, follow up

        main = F.interpolate(main, scale_factor=2, mode="bilinear", align_corners=True)     # 1. up first, follow conv
        # main = self.main_conv1(main)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = self.out_prelu(main + ext)

        return out


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 dropout_prob=0., bias=False, non_linear='ReLU'):
        super(DilatedBlock, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, kernel_size,
                               stride, padding=int((kernel_size - 1) / 2 * dilation), dilation=dilation, groups=1,
                               bias=bias)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=bias)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu(self.conv1_bn(self.conv1(x)))
        main = self.relu(self.conv2_bn(self.conv2(main)))
        main = self.conv4_bn(self.conv4(main))
        main = self.regul(main)
        out = self.relu(torch.add(main, residual))
        return out


class Factorized_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 dropout_prob=0., bias=False, non_linear='ReLU'):
        super(Factorized_Block, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        self.compress_conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, padding=0, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.conv2_1 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size, 1), stride=(stride, 1),
                                 padding=(int((kernel_size - 1) / 2 * dilation), 0), dilation=(dilation, 1), bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2_2 = nn.Conv2d(self.internal_channels, self.internal_channels, (1, kernel_size), stride=(1, stride),
                                 padding=(0, int((kernel_size - 1) / 2 * dilation)), dilation=(1, dilation), bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.extend_conv3 = nn.Conv2d(self.internal_channels, out_channels, 1, padding=0, bias=bias)

        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu((self.conv1_bn(self.compress_conv1(x))))
        main = self.relu(self.conv2_1_bn(self.conv2_1(main)))
        main = self.relu(self.conv2_2_bn(self.conv2_2(main)))

        main = self.conv3_bn(self.extend_conv3(main))
        main = self.regul(main)
        out = self.relu((torch.add(residual, main)))
        return out


class FSSNet(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.initial_block = InitialBlock(3, 16)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1, dropout_prob=0.03)
        self.factorized1_1 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_2 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_3 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_4 = Factorized_Block(64, 64, dropout_prob=0.03)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, dropout_prob=0.3)
        self.dilated2_1 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_2 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_3 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.dilated2_4 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_5 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_6 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.3)
        self.bottleneck4_1 = DilatedBlock(64, 64, dropout_prob=0.3)
        self.bottleneck4_2 = DilatedBlock(64, 64, dropout_prob=0.3)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.3)
        self.bottleneck5_1 = DilatedBlock(16, 16, dropout_prob=0.3)
        self.bottleneck5_2 = DilatedBlock(16, 16, dropout_prob=0.3)

        self.transposed_conv = nn.ConvTranspose2d(16, classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # Initial block
        # Initial block
        x = self.initial_block(x)

        # Encoder - Block 1
        x_1= self.downsample1_0(x)
        x = self.factorized1_1(x_1)
        x = self.factorized1_2(x)
        x = self.factorized1_3(x)
        x = self.factorized1_4(x)

        # Encoder - Block 2
        x_2 = self.downsample2_0(x)
        # print(x_2.shape)
        x = self.dilated2_1(x_2)
        x = self.dilated2_2(x)
        x = self.dilated2_3(x)
        x = self.dilated2_4(x)
        x = self.dilated2_5(x)
        x = self.dilated2_6(x)
        # print(x.shape)

        # Decoder - Block 3
        x = self.upsample4_0(x, x_2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # Decoder - Block 4
        x = self.upsample5_0(x, x_1)
        x = self.bottleneck5_1(x)
        x = self.bottleneck5_2(x)

        # Fullconv - DeConv
        x = self.transposed_conv(x)

        return x


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSSNet(classes=19).to(device)
    summary(model,(3,512,1024))
