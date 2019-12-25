import sys
import argparse
import torch


from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FPENet import FPENet






from tools.flops_counter.ptflops import get_model_complexity_info



pt_models = {

    'ENet': ENet,
    'ERFNet': ERFNet,
    'CGNet': CGNet,
    'EDANet': EDANet,
    'ESNet': ESNet,
    'ESPNet': ESPNet,
    'LEDNet': LEDNet,
    'EESPNet_Seg': EESPNet_Seg,
    'FastSCNN': FastSCNN,
    'DABNet': DABNet,
    'FPENet': FPENet
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='ENet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        net = pt_models[args.model](classes=19).cuda()

        flops, params = get_model_complexity_info(net, (3, 512, 1024),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        print('Flops: ' + flops)
        print('Params: ' + params)