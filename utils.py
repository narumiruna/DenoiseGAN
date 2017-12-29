import os

import numpy as np
import torch
from numpy.random import poisson
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def avg_psnr(x, y):
    batch_size, _, _, _ = x.shape
    s = 0
    for xx, yy in zip(x, y):
        s += psnr(xx, yy)
    return s / batch_size


def psnr(x, y):
    # TD-DO: fix psnr bug
    """
    x: ground turth
    y: noisy image
    """
    mse = np.array((x - y)**2).mean()
    max_x = 1.0
    psnr = 10 * np.log10(max_x**2 / mse)
    # print('psnr: {}, mse: {}, max_x: {}'.format(psnr, mse, max_x))
    return psnr


def var_to_numpy(var):
    return var.cpu().data.numpy()


def poisson_noise(image, peak=30):
    image = np.array(image)
    ratio = peak / 255.0
    output = poisson(image * ratio) / ratio
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = Image.fromarray(output)
    return output


class NoiseDataset(data.Dataset):
    def __init__(self, root, transform=None, size=10):
        super(NoiseDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.random_crop = transforms.RandomCrop(size)

        self.image_dirs = [d for d in os.listdir(
            root) if os.path.isdir(os.path.join(root, d))]

        self.image_paths = []
        for image_dir in self.image_dirs:
            image_filenames = os.listdir(os.path.join(root, image_dir))
            for image_filename in image_filenames:
                if image_filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(
                        self.root, image_dir, image_filename))

    def __getitem__(self, index):
        # load image
        image = datasets.folder.default_loader(self.image_paths[index])

        # crop image
        image = self.random_crop(image)

        # add poisson noise
        noisy = poisson_noise(image)

        if self.transform:
            image = self.transform(image)
            noisy = self.transform(noisy)

        return image, noisy

    def __len__(self):
        return len(self.image_paths)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
    ])
    dataset = NoiseDataset('data', transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=64)

    for i, (x, y) in enumerate(dataloader):
        print(x.numpy().mean(), x.numpy().std(),
              x.numpy().max(), x.numpy().min())


if __name__ == '__main__':
    main()
