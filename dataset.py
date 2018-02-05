import glob
import os

import numpy as np
import torch
from numpy.random import poisson
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader

from random import randint


class RENOIR(data.Dataset):
    def __init__(self, root, crop_size=64, transform=None):
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(self.root, '*/*/*Noisy.bmp')))

    def __getitem__(self, index):
        noisy_path = self.paths[index]
        ref_path = glob.glob(os.path.join(*(noisy_path.split('/')[:-1]), '*Reference.bmp'))[0]

        # 一張 3000x3000 這樣做效率很差, 不過先這樣以後再說
        noisy = pil_loader(noisy_path)
        ref = pil_loader(ref_path)

        # crop
        w, h = noisy.size
        size = self.crop_size

        left, upper = randint(0, w - size), randint(0, h - size)
        box = (left,  upper, left + size, upper + size)

        crop_noisy = noisy.crop(box)
        crop_ref = noisy.crop(box)

        if self.transform:
            crop_noisy = self.transform(crop_noisy)
            crop_ref = self.transform(crop_ref)

        return crop_noisy, crop_ref

    def __len__(self):
        return len(self.paths)


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dst = RENOIR('data/renoir', transform=transform)
    loader = data.DataLoader(dst, batch_size=50, shuffle=False)
    for i, (x, y) in enumerate(loader):
        print(x.size())


if __name__ == '__main__':
    main()


def poisson_noise(image, peak=30):
    image = np.array(image)
    ratio = peak / 255.0
    output = poisson(image * ratio) / ratio

    # convex combination
    t = np.random.rand()
    output = t * image + (1 - t) * output

    output = np.clip(output, 0, 255).astype(np.uint8)
    output = Image.fromarray(output)
    return output


class NoisyCoco(data.Dataset):
    def __init__(self, root, transform=None, crop_size=128):
        super(NoisyCoco, self).__init__()
        self.root = root
        self.crop_size = crop_size
        self.random_crop = transforms.RandomCrop(crop_size)
        self.paths = glob.glob(os.path.join(root, '*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        image = pil_loader(path)

        # crop
        image = self.random_crop(image)

        # add poisson noise
        noisy = poisson_noise(image)

        if self.transform:
            image = self.transform(image)
            noisy = self.transform(noisy)

        return noisy, image

    def __len__(self):
        return len(self.paths)
