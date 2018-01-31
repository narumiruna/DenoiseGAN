import requests

import wget
from bs4 import BeautifulSoup
import io
from PIL import Image

RAW_HEADERS = """Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8
Accept-Encoding:gzip, deflate
Accept-Language:zh-TW,zh;q=0.9,ja;q=0.8,en-US;q=0.7,en;q=0.6
Cache-Control:max-age=0
Connection:keep-alive
Cookie:__cfduid=d00174e86a16ba191d07dea89c7163ff61517387425
DNT:1
Host:www.insecam.org
Referer:http://www.insecam.org/en/byrating/
Upgrade-Insecure-Requests:1
User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"""

class Insecam:
    def __init__(self, url):
        self.sess = requests.session()
        self.url = url
        self.headers = {line.partition(':')[0]: line.partition(':')[2] for line in RAW_HEADERS.split("\n")}

    def get_frame(self):
        res = self.sess.get(self.url, headers=self.headers)

        # find image src
        soup = BeautifulSoup(res.text, 'html.parser')
        img = soup.find('img', {'id': 'image0'})
        img_src = img.get('src')

        # convert to pil image
        img_res = self.sess.get(img_src)
        pil_img = Image.open(io.BytesIO(img_res.content)).convert('RGB')

        return pil_img

    def close(self):
        self.sess.close()

if __name__ == '__main__':
    url = 'http://www.insecam.org/en/view/506886/'
    cam = Insecam(url)
    cam.get_frame().show()
    cam.close()