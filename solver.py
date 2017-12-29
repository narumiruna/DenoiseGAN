import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

from model import DeepClassAwareDenoiseNet
from utils import NoiseDataset, var_to_numpy, avg_psnr


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

        self.optimizer = torch.optim.Adam(self.net.parameters(), args.learning_rate)

        self.best_loss = float('Inf')

    def solve(self):
        for epoch in range(self.args.epochs):
            self.train(epoch)


    def save_model(self, epoch, index):
        filename = os.path.join(self.args.model_dir, 'model_{}_{}.pth'.format(epoch, index))
        print('Saving model {}'.format(filename))
        torch.save(self.net.state_dict(), filename)

    def train(self, epoch):
        for i, (image, noisy) in enumerate(self.dataloader):
            image = Variable(image)
            noisy = Variable(noisy)

            if self.args.cuda:
                image = image.cuda()
                noisy = noisy.cuda()

            loss = (self.net(image) - image).pow(2).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 100 == 0:
                print('Train epoch: {}, loss: {}, psnr: {}.'.format(epoch,
                                                                    float(var_to_numpy(loss)),
                                                                    avg_psnr(var_to_numpy(image), var_to_numpy(noisy))))
                cur_loss = float(var_to_numpy(loss))
                if cur_loss < self.best_loss:
                    self.best_loss = cur_loss
                    self.save_model(epoch, i)

    def evaluate(self):
        pass

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = NoiseDataset('data', transform=transform)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True,
                                     num_workers=2)

        return dataloader
