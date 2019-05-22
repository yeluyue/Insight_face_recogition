# coding: utf-8
import mxnet as mx
import cv2
import os
import time
import numpy as np
from skimage import transform as trans
from mtcnn import MTCNN
from PIL import Image



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

    mtcnn = MTCNN()

    dataset_path = '/home/yeluyue/yeluyue/DataSet/lfw/lfw_all'
    img_ext = '.jpg'
    imgs_list = list_img_name(dataset_path, img_ext)

    dataset_path_save = '/home/yeluyue/yeluyue/DataSet/lfw/lfw_112x112_mtcnn'
    people_num = 0
    img_num = 0

    for img_path in imgs_list:

        img_save_folder = os.path.join(dataset_path_save, img_path.rsplit('/')[-2])
        if not os.path.exists(img_save_folder):
            os.mkdir(img_save_folder)
            people_num += 1
        img_name_jpg = img_path.rsplit('/')[-1]
        img_name_png = img_name_jpg.replace('jpg', 'png')
        img_path_save = os.path.join(img_save_folder, img_name_png)
        print(img_path_save)

        img = Image.open('/home/yeluyue/yeluyue/DataSet/lfw/lfw_all/Abdullah_Gul/Abdullah_Gul_0016.jpg')
        if img.layers == 2:
            img_extend = 2

        img_align = mtcnn.align(img)
        img_align.save(img_path_save)
        img_num += 1
        print('peolpe_num:', people_num)
        print('img_num:', img_num)



