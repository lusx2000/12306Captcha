import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fastai.vision import *
from fastai.metrics import error_rate, accuracy

def get_vertical_hist(src):
    # vertical projection histogram
    # input binary np array
    binary = (src == 255)
    hist = np.sum(binary, axis=0)
    return hist


def get_all_images(src):
    # Single image size: 66x66 px
    # gap between every image 5px
    # top left image point (5, 41)

    sub1 = src[41:41 + 66, 5 + 72*0:5 + 72*0 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub2 = src[41:41 + 66, 5 + 72*1:5 + 72*1 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub3 = src[41:41 + 66, 5 + 72*2:5 + 72*2 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub4 = src[41:41 + 66, 5 + 72*3:5 + 72*3 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub5 = src[113:113 + 66, 5 + 72*0:5 + 72*0 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub6 = src[113:113 + 66, 5 + 72*1:5 + 72*1 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub7 = src[113:113 + 66, 5 + 72*2:5 + 72*2 + 66].transpose([2, 0, 1]).astype(np.float32)/255
    sub8 = src[113:113 + 66, 5 + 72*3:5 + 72*3 + 66].transpose([2, 0, 1]).astype(np.float32)/255

    s1 = torch.from_numpy(sub1)
    s2 = torch.from_numpy(sub2)
    s3 = torch.from_numpy(sub3)
    s4 = torch.from_numpy(sub4)
    s5 = torch.from_numpy(sub5)
    s6 = torch.from_numpy(sub6)
    s7 = torch.from_numpy(sub7)
    s8 = torch.from_numpy(sub8)

    return s1, s2, s3, s4, s5, s6, s7, s8


def get_char_images(src):
    # char img start (118, 0) (118, 28) length:29
    # two types : one vocab with light background
    #             two vocabs with dark background
    ch_img = src[0:28, 118:290]
    binary = cv2.cvtColor(ch_img, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(binary,120, 255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print(binary.sum())
    # single vocal
    if (binary.sum() <= 200000):
        hist = get_vertical_hist(binary)
        for i in range(len(hist)-1, 0, -1):
            if hist[i] != 0:
                out = ch_img[0:28, 0:i]
                out = out.transpose([2, 0, 1]).astype(np.float32)/255
                out = torch.from_numpy(out)
                break
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        plt.imshow(binary)
        hist = get_vertical_hist(binary)
        split = []
        for i in range(len(hist)-1, 0, -1):
            if hist[i] == 0 and hist[i-1] != 0:
                split.append(i)
        # print(split)
        try:
            assert len(split) == 2
            c1 = ch_img[0:28, 0: split[1]]
            c2 = ch_img[0:28, split[1]: split[0]]
        except:
            c1 = ch_img[0:28, 0: split[-1]]
            c2 = ch_img[0:28, split[-1]: split[0]]
        c1 = c1.transpose([2, 0, 1]).astype(np.float32)/255
        c2 = c2.transpose([2, 0, 1]).astype(np.float32)/255

        c1 = torch.from_numpy(c1)
        c2 = torch.from_numpy(c2)

        out = [c1, c2]
    return out


def inference_one(img):
    path = Path('/data/12306/images/')
    data = ImageDataBunch.from_folder(path,
                                  valid_pct=0.2).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet34,
                    metrics=[error_rate, accuracy])
    learn.load('stage-1')

    out = learn.predict(Image(img))

    return out
