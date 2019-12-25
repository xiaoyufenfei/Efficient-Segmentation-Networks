import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.convert_state import convert_state_dict


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=2, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./server/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args



def predict(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    for i, (input, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # Save the predict greyscale output for Cityscapes official evaluation
        # Modify image name to meet official requirement
        name[0] = name[0].rsplit('_', 1)[0] + '*'
        save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                     output_grey=True, output_color=False, gt_color=False)


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning testing")
    print("test set length: ", len(testLoader))
    predict(args, testLoader, model)


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
