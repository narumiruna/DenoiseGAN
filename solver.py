import os
from math import log10
from time import time

import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image

from dataset import NoisyCoco
from model import DeepClassAwareDenoiseNet


from utils import psnr


class Solver(object):
    def __init__(self, args, net, dataloader):
        self.args = args
        self.dataloader = dataloader
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
            t = Variable(torch.rand(1))

            if self.args.cuda:
                noisy = noisy.cuda()
                image = image.cuda()
                t = t.cuda()

            noisy = t * image + (1 - t) * noisy

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
                                                                           'loss', cur_loss,
                                                                           'noisy psnr', noisy_psnr,
                                                                           'denoised psnr', denoised_psnr))
                # save current training results
                path = os.path.join(self.args.model_dir, 'image_{}_{}.jpg'.format(epoch, i))
                print('Saving image {}'.format(path))
                cat_tensor = torch.cat([image.data, noisy.data, denoised.data])
                save_image(cat_tensor, path, nrow=self.args.batch_size)

                # save best model
                if cur_loss !=  self.best_loss:
                    self.save_model(epoch, i)