import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


from torch import nn


class DeepClassAwareDenoiseNet(nn.Module):
    def __init__(self, in_ch, c_ch, n_layers):
        super(DeepClassAwareDenoiseNet, self).__init__()
        self.in_ch = in_ch
        self.c_ch = c_ch

        layers = [nn.Conv2d(in_ch, c_ch, 3, 1, 1)]
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c_ch - in_ch, c_ch, 3, 1, 1))
        self.module_list = nn.ModuleList(layers)

    def forward(self, input_):
        output = input_

        main = input_
        for i, layer in enumerate(self.module_list):
            main = layer(main)
            output = output + main[:, :self.in_ch]
            main = main[:, self.in_ch:]
            if i < len(self.module_list) - 1:
                main = F.leaky_relu(main, negative_slope=0.2)

        return output


class Discriminator(nn.Module):
    def __init__(self, in_height, in_width):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU()
        )

        self.linear = nn.Linear(4 * in_height * in_width, 1)

    def forward(self, input_):
        output = self.conv(input_)
        print(output.size())
        output = self.linear(output)

        return output


def main():
    n = DeepClassAwareDenoiseNet(3, 12, 8)
    # d = Discriminator(64, 64)
    x = Variable(torch.randn(64, 3, 64, 64))
    print(n(x))
    # z = d(n(x))
    # print(z)


if __name__ == '__main__':
    main()
    # n = Net()