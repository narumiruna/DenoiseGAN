import argparse

from solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=2)
    parser.add_argument('-lr', type=float, default=1e-3)
    args = parser.parse_args()

    solver = Solver(args)
    solver.solve()


if __name__ == '__main__':
    main()
