

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

from model import DeepClassAwareDenoiseNet
from utils import NoiseDataset, var_to_numpy


class Solver(object):
    def __init__(self, config):
        self.config = config
        self.dataloader = self.get_dataloader()
        self.net = DeepClassAwareDenoiseNet(3, 6, 3)

        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr)

    def solve(self):
        for epoch in range(20):
            self.train(epoch)

    def train(self, epoch):
        for i, (image, noisy) in enumerate(self.dataloader):
            image = Variable(image)
            noisy = Variable(noisy)

            if torch.cuda.is_available():
                image = image.cuda()
                noisy = noisy.cuda()

            denoised = self.net(image)
            loss = (denoised - image).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 1000 == 0:
                print('Train epoch: {}, loss: {}.'.format(
                    epoch, var_to_numpy(loss)))

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = NoiseDataset('data', transform=transform)
        dataloader = data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True)

        return dataloader
