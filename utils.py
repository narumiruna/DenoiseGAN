from icrawler.builtin  import GoogleImageCrawler
from torch.utils import data
from PIL import Image
import glob
import os
from torchvision import transforms
import torch
from torchvision import datasets

datasets.MNIST

transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.ToTensor()
])





class GoogleImageDataset(data.Dataset):
    def __init__(self, root, keyword, train=True, transform=None, target_transform=None, download=False):
        super(GoogleImageDataset, self).__init__()
        self.root = root
        self.keyword=keyword
        self.train_data = None

        self.max_num = 1000

    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

    def download(self):
        google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                        storage={'root_dir': os.path.join(self.root, self.keyword)})
        google_crawler.crawl(self.keyword, max_num=self.max_num,
                            date_min=None, date_max=None,
                            min_size=(200,200), max_size=None)

    def load(self):
        pattern='*.{}'.format(format)
        paths = glob.glob(os.path.join(self.root, self.keyword,pattern))

        images = []
        for path in paths:
            img = transform(Image.open(path))
            if img.size()[0] == 3:
                images.append(img)
        self.train_data = torch.stack(images)

def main():
    dataset = GoogleImageDataset('data', 'おっぱい')
    dataset.download()
    dataset.load()

if __name__ == '__main__':
    main()