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
    def __init__(self, args, net, train_dataloader, val_dataloader):
        self.args = args
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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
        for i, (noisy, image) in enumerate(self.train_dataloader):
            self.net.train()

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
            # if cur_loss < self.best_loss:
            #     self.best_loss = cur_loss
            #     self.best_state_dict = self.net.state_dict()

            if i % self.args.log_interval == 0:
                avg_val_loss = self.evaluate()
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self.best_state_dict = self.net.state_dict()

                noisy_psnr = psnr(image.data, noisy.data)
                denoised_psnr = psnr(image.data, denoised.data)
                print('{}: {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}.'.format('Train epoch', epoch,
                                                                           'avg_val_loss', avg_val_loss,
                                                                           'noisy psnr', noisy_psnr,
                                                                           'denoised psnr', denoised_psnr))
                # save current training results
                path = os.path.join(self.args.image_dir, '{}_{}.jpg'.format(epoch, i))
                print('Saving image {}'.format(path))
                cat_tensor = torch.cat([image.data, noisy.data, denoised.data])
                save_image(cat_tensor, path, nrow=self.args.batch_size)

                # save best model
                self.save_model(epoch, i)

    def evaluate(self, max_batch=10):
        self.net.eval()

        losses = []
        for i, (noisy, image) in enumerate(self.val_dataloader):
            noisy = Variable(noisy, volatile=True)
            image = Variable(image, volatile=True)

            if self.args.cuda:
                noisy = noisy.cuda()
                image = image.cuda()

            denoised = self.net(noisy)
            loss = (denoised - image).pow(2).mean()

            losses.append(loss.data)

            if not i < max_batch:
                break

        avg_loss = torch.cat(losses).mean()
        print('avg loss: {}'.format(avg_loss))
        return avg_loss