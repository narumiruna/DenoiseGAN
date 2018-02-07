import argparse
import torch
from solver import Solver
import os
from model import DeepClassAwareDenoiseNet
from torch.utils import data
from utils import get_transform
from dataset import NoisyCoco, RENOIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--layers', '-l', type=int, default=20)
    parser.add_argument('--channels', '-ch', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--crop-size', type=int, default=64)
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.makedirs(args.model_dir, exist_ok=True)

    print(args)

    net = DeepClassAwareDenoiseNet(3, args.channels, args.layers)

    # dataloader = data.DataLoader(RENOIR(root='data/renoir',
    #                                         transform=get_transform(),
    #                                         crop_size=args.crop_size),
    #                                 batch_size=args.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=args.workers)

    dataloader = data.DataLoader(NoisyCoco(root='data/coco/train2017',
                                            transform=get_transform(),
                                            crop_size=args.crop_size),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers)

    solver = Solver(args, net, dataloader)
    solver.solve()

if __name__ == '__main__':
    main()
