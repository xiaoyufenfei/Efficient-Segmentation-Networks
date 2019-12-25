######################################################################################
#LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
#Paper-Link: https://arxiv.org/abs/1905.02423
######################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  



__all__ = ["LEDNet"]

def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2 


def Merge(x1,x2):
    return torch.cat((x1,x2),1) 
    

def Channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    #reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    #flatten
    x = x.view(batchsize,-1,height,width)
    
    return x


class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output



class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output



# class Interpolate(nn.Module):
#     def __init__(self,size,mode):
#         super(Interpolate,self).__init__()
#         self.size = size
#         self.mode = mode
#     def forward(self,x):
#         x = F.interpolate(x,size=self.size,mode=self.mode,align_corners=True)
#         return x

        

class SS_nbt_module_paper(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        oup_inc = chann//2
        
        #dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        #dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)       
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

        # self.channel_shuffle = PermutationBlock(2)
       
    
    def forward(self, x):
    
        residual = x
    
        x1, x2 = Split(x)
    
        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1_mid = self.relu(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2_mid = self.relu(output2)

        output1 = self.conv3x1_2_l(output1_mid)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
      
        output2 = self.conv1x3_2_r(output2_mid)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = Merge(output1, output2)
        
        out = F.relu(residual + out)

        # out = self.channel_shuffle(out)   ### channel shuffle
        out = Channel_shuffle(out,2)   ### channel shuffle

        return out

        # return    ### channel shuffle


class APNModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APNModule, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )


        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch,1,kernel_size=(7,1),stride=(2,1),padding=(3,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,7),stride=(1,2),padding=(0,3),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True)
        )


        self.down2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(5,1),stride=(2,1),padding=(2,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,5),stride=(1,2),padding=(0,2),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(3,1),stride=(2,1),padding=(1,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,3),stride=(1,2),padding=(0,1),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(1,1,kernel_size=(3,1),stride=1,padding=(1,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,3),stride=1,padding=(0,1),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(5,1),stride=1,padding=(2,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,5),stride=1,padding=(0,2),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(7,1),stride=1,padding=(3,0),bias=True),
            nn.Conv2d(1,1,kernel_size=(1,7),stride=1,padding=(0,3),bias=True),
            nn.BatchNorm2d(1, eps=1e-03),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        h,w = x.size()[2:]

        b1 = self.branch1(x)
        b1= F.interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3= F.interpolate(x3, size=((h+3) // 4, (w+3) // 4), mode="bilinear", align_corners=True)

        x2 = self.conv2(x2)
        x = x2 + x3
        x= F.interpolate(x, size=((h+1) // 2, (w+1) // 2), mode="bilinear", align_corners=True)


        x1 = self.conv1(x1)
        x = x + x1
        x= F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        x = torch.mul(x, mid)

        x = x + b1

        return x


class LEDNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,32)
        
        self.layers = nn.ModuleList()

        for x in range(0, 3):   
           self.layers.append(SS_nbt_module_paper(32, 0.03, 1)) 
        

        self.layers.append(DownsamplerBlock(32,64))
        

        for x in range(0, 2):   
           self.layers.append(SS_nbt_module_paper(64, 0.03, 1)) 
  
        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    
            self.layers.append(SS_nbt_module_paper(128, 0.3, 1))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 2))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 5))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 9))
            
        for x in range(0, 1):    
            self.layers.append(SS_nbt_module_paper(128, 0.3, 2))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 5))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 9))
            self.layers.append(SS_nbt_module_paper(128, 0.3, 17))
                    
        self.apn = APNModule(in_ch=128,out_ch=classes)

        #self.output_conv = nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        #self.output_conv = nn.ConvTranspose2d(128, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        #self.output_conv = nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
            
        output = self.apn(output)
        out = F.interpolate(output, input.size()[2:], mode="bilinear", align_corners=True)
        # print(out.shape)

        return out

         

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LEDNet(classes=19).to(device)
    summary(model,(3,360,480))
