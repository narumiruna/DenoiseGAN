from torchvision import transforms
from math import log10

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def psnr(x, y, pixel_max=1.0):
    x = (x + 1) / 2
    y = (y + 1) / 2

    mse = (x - y).pow(2).mean()
    if mse == 0:
        return 100

    return 10 * log10(pixel_max ** 2 / float(mse))