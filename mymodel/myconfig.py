import time
from argparse import ArgumentParser
import torch
from torch.backends import cudnn


# some configs prepare


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

def args_info():
    parser = ArgumentParser(description='OCM')

    parser.add_argument('--data-dir', type=str, default='../ocm-master/data', help="path to dataset")

    parser.add_argument('--polyvore-split', default='nondisjoint', type=str, choices=['nondisjoint', 'disjoint'], help="version of dataset")

    parser.add_argument('-epoch', type=int, default=5) 

    parser.add_argument('-bs', dest='batch_size', type=int, default= 64, help="batch size")#64

    parser.add_argument('-j', dest='num_worker', type=int, default=0, help="number of worker")

    # hid for images feature
    parser.add_argument('-hid', type=int, default=64, help="the size of image embedding")

    parser.add_argument('-lr', type=float, default=5e-5, help="learning rate")

    args = parser.parse_args()
    _ = print("=" * 15, "args", "=" * 15), print(args), print("=" * 36)
    return args
