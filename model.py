import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class DenoiseNet(nn.Module):
    def __init__(self, in_channels, conv_channels, num_layers):
        super(DenoiseNet, self).__init__()

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_layers = num_layers

        self.input_layer = nn.Conv2d(in_channels, conv_channels, 3, 1, 1)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(conv_channels, conv_channels, 3, 1, 1))

    def forward(self, input_):
        cur_layer = self.input_layer(input_)
        output = input_ + cur_layer[:, 0:self.in_channels, :, :]
        cur_layer = F.relu(cur_layer)

        for i, layer in enumerate(self.layers):
            cur_layer = layer(cur_layer)
            output += cur_layer[:, 0:self.in_channels, :, :]
            if i < self.num_layers - 2:
                cur_layer = F.relu(cur_layer)

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

        self.linear = nn.Linear(4*in_height*in_width, 1)


    def forward(self, input_):
        output = self.conv(input_)
        output = self.linear(output)


        return output

def main():
    n = DenoiseNet(3, 64, 20)
    d = Discriminator(64,64)
    # n.cuda()
    # d.cuda()
    x = Variable(torch.randn(64, 3, 64,64))
    # x = x.cuda()
    print(n(x))
    z = d(n(x))
    print(z)


if __name__ == '__main__':
    main()