import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms

from utils import ImageFolder, psnr

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--batch_size', '-bs', type=int, default=128)
parser.add_argument('-lr', type=float, default=1e-4)
args = parser.parse_args()


class BasicBlock(nn.Module):
    def __init__(self, in_ch=63, out_ch=64):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch - 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, 1, 3, 1, 1)

    def forward(self, input_):
        main = F.relu(self.conv1(input_))
        residual = self.conv2(input_)
        return main, residual


class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()

        self.layer1 = BasicBlock(1, 10)
        self.layer2 = BasicBlock(9, 10)
        self.layer5 = nn.Conv2d(9, 1, 3, 1, 1)

    def forward(self, input_):
        output = input_
        # first layer
        main, residual = self.layer1(input_)
        output += residual
        # secibd layer
        main, residual = self.layer2(main)
        output += residual
        # last layer
        output += self.layer5(main)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 1, 0),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 29 * 29, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, input_):
        output = self.conv(input_)
        output = output.view(-1, 64 * 29 * 29)
        output = self.linear(output)
        return output


train_dataset = ImageFolder('data')
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size)

n = DenoiseNet()
d = Discriminator()

if args.cuda:
    n.cuda()
    d.cuda()

optimizer_n = torch.optim.Adam(n.parameters(), lr=args.lr)
optimizer_d = torch.optim.Adam(d.parameters(), lr=args.lr)


def train(epoch):
    for batch_idx, (noisy, real) in enumerate(train_dataloader):
        noisy = Variable(noisy)
        real = Variable(real)

        if args.cuda:
            noisy = noisy.cuda()
            real = real.cuda()

        denoised = n(noisy)
        loss_d = -d(real).mean() + d(denoised).mean()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step

        if batch_idx % 10 == 0:
            loss_n = -d(denoised).mean()

            optimizer_n.zero_grad()
            loss_n.backward()
            optimizer_n.step()

        print(batch_idx)

    gt = real.cpu().data.numpy()
    pred = denoised.cpu().data.numpy()
    print('psnr: {}'.format(psnr(gt[0], pred[0])))

def main():
    train(0)


if __name__ == '__main__':
    main()
