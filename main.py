import argparse
import torch
from solver import Solver
import os
from model import DeepClassAwareDenoiseNet
from torch.utils import data
from utils import get_transform
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--n-channels', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--train-dir', type=str, default='data/train2017')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    print(args)

    net = DeepClassAwareDenoiseNet(3, args.n_channels, args.n_layers)

    dataloader = data.DataLoader(NoisyCoco(root=args.train_dir,
                                            transform=get_transform(),
                                            crop_size=args.crop_size),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)

    solver = Solver(args, net, dataloader)
    solver.solve()

if __name__ == '__main__':
    main()
