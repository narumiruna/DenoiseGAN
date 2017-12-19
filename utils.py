import glob
import os

import torch
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms


class GoogleImage(data.Dataset):
    def __init__(self,
                 root,
                 keyword,
                 transform=None,
                 download=False,
                 min_size=[200, 200],
                 max_num=10000,
                 image_format='jpg'
                 ):
        super(GoogleImage, self).__init__()
        self.root = root
        self.keyword = keyword
        self.max_num = max_num
        self.min_size = min_size
        self.transform = transform
        self.image_format = image_format
        if download:
            self.download()

        self.train_data = self.load()

    def __getitem__(self, index):
        img = self.train_data[index]
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.train_data)

    def download(self):
        google_crawler = GoogleImageCrawler(parser_threads=2,
                                            downloader_threads=4,
                                            storage={'root_dir': os.path.join(self.root, self.keyword)})
        google_crawler.crawl(self.keyword,
                             max_num=self.max_num,
                             date_min=None, date_max=None,
                             min_size=self.min_size,
                             max_size=None)

    def load(self):
        pattern = '*.{}'.format(self.image_format)
        paths = glob.glob(os.path.join(self.root, self.keyword, pattern))
        if not paths:
            raise Exception('There is no image file.')
        images = []
        for path in paths:
            image = Image.open(path)
            # append RGB image
            if image.mode == 'RGB':
                images.append(image)
        return images

def main():
    dataset = GoogleImage('data', 'tree', download=True, max_num=100)

if __name__ == '__main__':
    main()
