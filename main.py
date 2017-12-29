import argparse
import torch
from solver import Solver
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_layers', type=int, default=20)
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.makedirs(args.model_dir, exist_ok=True)

    print(args)

    solver = Solver(args)
    solver.solve()


if __name__ == '__main__':
    main()
