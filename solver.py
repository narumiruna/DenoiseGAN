import os
from math import log10
from time import time

import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.utils import make_grid, save_image

from dataset import NoisyCoco
from model import DeepClassAwareDenoiseNet


def psnr(x, y, pixel_max=1.0):
    # shift
    x = (x + 1) / 2
    y = (y + 1) / 2

    mse = (x - y).pow(2).mean()
    if mse == 0:
        return 100

    return 10 * log10(pixel_max ** 2 / float(mse))


class Solver(object):
    def __init__(self, args, net):
        self.args = args
        self.dataloader = self.get_dataloader()
        self.net = net

        if args.parallel:
            print('Enable data parallel.')
            self.net = nn.DataParallel(self.net)

        if args.cuda:
            print('Enable cuda.')
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          args.learning_rate)

        self.best_loss = float('Inf')
        self.best_state_dict = None

    def solve(self):
        for epoch in range(self.args.epochs):
            self.train(epoch)

    def save_model(self, epoch, index):
        filename = os.path.join(self.args.model_dir,
                                'model_{}_{}.pth'.format(epoch, index))
        print('Saving model {}'.format(filename))

        state_dict = self.best_state_dict
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()

        torch.save(state_dict, filename)

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

            cur_loss = float(loss.data)
            if cur_loss < self.best_loss:
                self.best_loss = cur_loss
                self.best_state_dict = self.net.state_dict()

            if i % self.args.log_interval == 0:
                noisy_psnr = psnr(image.data, noisy.data)
                denoised_psnr = psnr(image.data, denoised.data)
                print('{}: {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}.'.format('Train epoch', epoch,
                                                                           'loss', float(
                                                                               loss.data),
                                                                           'noisy psnr', noisy_psnr,
                                                                           'denoised psnr', denoised_psnr))
                print('Saving images...')
                save_image(torch.cat([image.data, noisy.data, denoised.data]),
                           os.path.join(self.args.image_dir,
                                        '{}_{}.jpg'.format(epoch, i)),
                           nrow=self.args.batch_size)

                self.save_model(epoch, i)

    def load_model(self, f):
        state_dict = torch.load(f)
        self.net.load_state_dict(state_dict)


    def evaluate(self):
        pass

    def predict(self, input_):
        pass

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataloader = data.DataLoader(NoisyCoco(root=self.args.train_dir,
                                               transform=transform,
                                               crop_size=self.args.crop_size),
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=self.args.num_workers)

        return dataloader
