import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from dataset import NoisyCoco
from model import DeepClassAwareDenoiseNet

def psnr(image_1, image_2, pixel_max=1.0):
    """
    Args:
        image_1: A 'numpy.ndarray' representing the first image.
        image_2: A 'numpy.ndarray' representing the second image.
        pixel_max: A value representing the maximum possible pixel value of the image.
    Returns:
        A value representing PSNR.
    """
    mse = np.power(image_1 - image_2, 2).mean()
    if mse == 0:
        return 100

    return 10 * np.log10(pixel_max **2 / mse)


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.dataloader = self.get_dataloader()
        self.net = DeepClassAwareDenoiseNet(3, args.n_channels, args.n_layers)

        if args.parallel:
            print('Enable data parallel.')
            self.net = nn.DataParallel(self.net)

        if args.cuda:
            print('Enable cuda.')
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          args.learning_rate)

        self.best_loss = float('Inf')

    def solve(self):
        for epoch in range(self.args.epochs):
            self.train(epoch)

    def save_model(self, epoch, index):
        filename = os.path.join(self.args.model_dir,
                                'model_{}_{}.pth'.format(epoch, index))
        print('Saving model {}'.format(filename))
        torch.save(self.net.state_dict(), filename)

    def train(self, epoch):
        for i, (noisy, image) in enumerate(self.dataloader):
            noisy = Variable(noisy)
            image = Variable(image)

            if self.args.cuda:
                noisy = noisy.cuda()
                image = image.cuda()

            denoised = self.net(noisy)
            loss = (denoised - image).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.args.log_interval == 0:
                print('Train epoch: {}, loss: {}.'.format(epoch,
                                                          float(loss.data)))

                print('Saving images...')
                save_image(torch.cat([image.data, noisy.data, denoised.data]),
                           'images/{}_{}.jpg'.format(epoch, i),
                           nrow=self.args.batch_size)
                print(psnr(image.data, noisy.data),
                      psnr(image.data, denoised.data))

                cur_loss = float(loss.data)
                if cur_loss < self.best_loss:
                    self.best_loss = cur_loss
                    self.save_model(epoch, i)

    def evaluate(self):
        pass

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataloader = data.DataLoader(NoisyCoco('data',
                                               transform=transform,crop_size=64),
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers)

        return dataloader
