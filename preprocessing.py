import glob
from PIL import Image
import os


def remove_small_image(image_dir, min_size=64):
    paths = glob.glob(os.path.join(image_dir, '*.jpg'))

    for path in paths:
        image = Image.open(path).convert('RGB')
        w, h = image.size
        if w < min_size or h < min_size:
            print('Remove {} with (w, h) = ({}, {})'.format(path, w, h))
            os.remove(path)
            

if __name__ == '__main__':
    remove_small_image('data/train2017')
    remove_small_image('data/test2017')
    remove_small_image('data/val2017')
