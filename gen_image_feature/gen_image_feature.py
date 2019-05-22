from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd
import torch
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

from ShuffleNet_V2.shuffleNet_V2 import ShuffleNetV2
from model import Backbone, Arcface, MobileFaceNet_22, Am_softmax, l2_norm

import PIL

image_shape = None
net = None
data_size = 1862120
emb_size = 512
use_flip = True


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = trans.functional.hflip(data[idx, :, :])

def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_feature(buffer, model):
    global emb_size
    test_transform = trans.Compose([
        # trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    embs = torch.zeros(len(buffer), 512)

    if use_flip:
        img_all_flip = torch.zeros(len(buffer)*2, 3, 112, 112).cuda()
    else:
        img_all = torch.zeros(len(buffer), 3, 112, 112).cuda()
    i = 0
    for item in buffer:

        j = i + len(buffer)
        img = PIL.Image.open(item)  # to rgb
        # img.show()
        if use_flip:
            mirror = trans.functional.hflip(img)
            mirror_ten = test_transform(mirror).unsqueeze(0).cuda()
            img_ten = test_transform(img).to(0).unsqueeze(0).cuda()
            img_all_flip[i, :, :, :] = img_ten
            img_all_flip[j, :, :, :] = mirror_ten

        else:
            img_tensor = test_transform(img).to(0).unsqueeze(0)
            # print(i, './.........................................................................................')
            # print(img_tensor)
            img_all[i, :, :, :] = img_tensor
        i += 1
    if use_flip:
        embs_all_flip = model(img_all_flip)
        embs_all = embs_all_flip[0::2] + embs_all_flip[1::2]
    else:
        embs_all = model(img_all)
    # if use_flip:
    #     embs = l2_norm(embs_all[0::2] + embs_all[1::2])
    # else:
    #     embs = embs_all
    # embs = sklearn.preprocessing.normalize(embs_all.cpu().detach().numpy())
    embs = embs_all.cpu().detach().numpy()

    return embs


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows, cols, cols * 4, 5))
        f.write(m.data)


def main(args):
    global image_shape
    global net

    print(args)
    ctx = []
    ctx.append(mx.gpu(0))

    cvd = 0

    image_shape = [int(x) for x in args.image_size.split(',')]

    # model = ShuffleNetV2().to(0)
    model = MobileFaceNet_22(512).to(0)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    features_all = None
    i = 0
    fstart = 0
    buffer = []
    for line in open(os.path.join(args.input, 'filelist.txt'), 'r'):
        if i % 10000 == 0:
            print("processing ", i)
        i += 1
        line = line.strip()
        image_path = os.path.join(args.input, line)
        buffer.append(image_path)
        if len(buffer) == args.batch_size:
            embedding = get_feature(buffer, model)
            buffer = []
            fend = fstart + len(embedding)
            if features_all is None:
                features_all = np.zeros((data_size, emb_size), dtype=np.float32)
            # print('writing', fstart, fend)
            features_all[fstart:fend, :] = embedding
            fstart = fend
    if len(buffer) > 0:
        embedding = get_feature(buffer, model)
        fend = fstart + embedding.shape[0]
        print('writing', fstart, fend)
        features_all[fstart:fend, :] = embedding
    write_bin(args.output, features_all)
    # os.system("bypy upload %s"%args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=80)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--input', type=str, help='the dir of filelist.txt',
                        default='/home/yeluyue/dl/Datasets/ICCV_challenge/val_data/iccv19-challenge-data')
    parser.add_argument('--output', type=str, help='',
                        default='/home/yeluyue/dl/Datasets/work_space/ICCV2019_workspace/Msceleb_clean_zhu_mobilefacenet22_0.4_lr_e-1/features/mobilefacenet_22_have_flip.bin')
    parser.add_argument('--model', type=str, help='',
                        default='/home/yeluyue/dl/Datasets/work_space/ICCV2019_workspace/Msceleb_clean_zhu_mobilefacenet22_0.4_lr_e-1/save/model_2019-05-14-09-57_accuracy:0.0_step:2123636_None.pth')
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    main(parse_arguments(sys.argv[1:]))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



