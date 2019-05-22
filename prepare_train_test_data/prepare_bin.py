# coding=utf-8

'''
1-将图片转化为数组并存为二进制文件
'''

import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))

def list_img_name(path,img_ext):
    img_list = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        for img_name in os.listdir(folder_path):
            if os.path.splitext(img_name)[-1] == img_ext:
                img_path = os.path.join(folder_path, img_name)
                img_list.append(img_path)
    return img_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Package LFW images')
    # general
    parser.add_argument('--data-dir', default='/home/yeluyue/yeluyue/DataSet/MsCeleb_clean_10k', help='')
    parser.add_argument('--image-size', type=str, default='112,112', help='')
    parser.add_argument('--output', default='/home/yeluyue/yeluyue/DataSet/MsCeleb_clean_10k_bin', help='path to save.')
    args = parser.parse_args()
    data_dir = args.data_dir
    image_size = [int(x) for x in args.image_size.split(',')]

    lfw_paths = list_img_name(data_dir, '.png')
    lfw_bins = []
    lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
    i = 0
    for path in lfw_paths:
      with open(path, 'rb') as fin:
        _bin = fin.read()
        lfw_bins.append(_bin)
        img = mx.image.imdecode(_bin)
        img = nd.transpose(img, axes=(2, 0, 1))
        lfw_data[i][:] = img
        i+=1
        if i%1000==0:
          print('loading lfw', i)
    outname = args.output +'/data.bin'

    with open(outname, 'wb') as f:
      pickle.dump((lfw_bins), f, protocol=pickle.HIGHEST_PROTOCOL)
