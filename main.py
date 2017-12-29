import argparse
import torch
from solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_layers', type=int, default=20)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args)
    solver.solve()


if __name__ == '__main__':
    main()
