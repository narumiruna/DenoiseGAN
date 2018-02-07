import cv2
from PIL import Image
import numpy as np
import os

url ='http://93.87.72.254:8090/mjpg/video.mjpg'
# url = 'http://83.155.186.129:8084/mjpg/video.mjpg'

cam = cv2.VideoCapture()
cam.open(url)

image_dir = 'stream'
os.makedirs(image_dir, exist_ok=True)

i = 0
while True:
    ret_val, img = cam.read()

    if ret_val:
        h, w, _ = img.shape

        cv2.imshow('web stream', img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img).convert('RGB')
        pil_img.save(os.path.join(image_dir, str(i).zfill(6)+'.jpg'))

        i += 1

    if cv2.waitKey(60) == 27:
        break
    cv2.destroyAllWindows()
