###################################################################################################
#ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network
#Paper-Link: https://arxiv.org/pdf/1811.11431.pdf
###################################################################################################


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model.ESPNet_v2.Model import EESPNet, EESP
from model.ESPNet_v2.cnn_utils import *


__all__ = ["EESPNet_Seg"]


class EESPNet_Seg(nn.Module):
    def __init__(self, classes=19, s=2, pretrained=None, gpus=1):
        super().__init__()
        classificationNet = EESPNet(classes=1000, s=s)
        if gpus >=1:
            classificationNet = nn.DataParallel(classificationNet)
            # print(classificationNet)
        # load the pretrained weights
        if pretrained:
            if not os.path.isfile(pretrained):
                print('Weight file does not exist. Training without pre-trained weights')
            print('Model initialized with pretrained weights')
            classificationNet.load_state_dict(torch.load(pretrained))

        self.net = classificationNet.module

        del classificationNet
        # delete last few layers
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        if s <=0.5:
            p = 0.1
        else:
            p=0.2

        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters, self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2*self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize //2, stride=1, k=4, r_lim=7),
                PSPModule(pspSize // 2, pspSize //2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), C(pspSize // 2, classes, 1, 1))
        self.act_l3 = BR(classes)
        self.project_l2 = CBR(self.net.level2_0.act.num_parameters + classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), C(self.net.level1.act.num_parameters + classes, classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, size=out_l3.size()[2:], mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        # if self.training:
        #     return F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True), self.hierarchicalUpsample(proj_merge_l3_bef_act)
        # else:
        #     return F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)
        output = F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)
        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EESPNet_Seg(classes=19, s=2).to(device)
    summary(model,(3,512,1024))