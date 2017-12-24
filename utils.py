import os

import numpy as np
import skimage
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def add_poisson_noise(image):
    noisy = skimage.util.noise.random_noise(np.array(image), mode='poisson')
    return Image.fromarray(np.uint8(noisy * 255))


def psnr(x, y):
    """
    x: ground turth
    y: noisy image
    """
    mse = np.array((x - y)**2).mean()
    max_x = np.max(x)
    psnr = 10 * np.log10(max_x**2 / mse)
    # print('psnr: {}, mse: {}, max_x: {}'.format(psnr, mse, max_x))
    return psnr

class ImageFolder(data.Dataset):
    def __init__(self, root, size=128):
        super(ImageFolder, self).__init__()
        self.root = root
        self.size = size
        self.classes = [d for d in os.listdir(
            root) if os.path.isdir(os.path.join(root, d))]

        self.real_images = []
        for class_ in self.classes:
            for dir_path, _, filenames in os.walk(os.path.join(root, class_)):
                for filename in filenames:
                    if is_image_file(filename):
                        self.real_images.append(
                            os.path.join(dir_path, filename))

    def __getitem__(self, index):
        # get crop function
        random_crop = transforms.RandomCrop(self.size)

        # get real image
        real_image = Image.open(self.real_images[index])

        # randomly crop real image
        real_image = random_crop(real_image)

        # to grayscale
        real_image = F.to_grayscale(real_image)

        # get noisy image
        noisy_image = add_poisson_noise(real_image)

        # to tensor image
        noisy_image = F.to_tensor(noisy_image)
        real_image = F.to_tensor(real_image)

        # normalize
        noisy_image = F.normalize(noisy_image, (0.5,), (0.5,))
        real_image = F.normalize(real_image, (0.5,), (0.5,))
        return noisy_image, real_image

    def __len__(self):
        return len(self.real_images)
