from pathlib import Path
from config import get_config
from data.data_pipe import load_bin
import argparse
import os
import mxnet as mx
from tqdm import tqdm
from PIL import Image, ImageFile
import mxnet.ndarray as nd
from torchvision import transforms as trans
import numpy as np
import cv2
"""
prepare the val set images
using: python3 -r faces_lfw_0.4_112x112
"""
def load_mx_rec(rec_path):
    save_path = rec_path + '/imgs_1'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path + '/train.idx'), str(rec_path + '/train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label[0])
        img = Image.fromarray(img)
        # img.show()
        transform_tensor = trans.ToTensor()
        transform_pil = trans.ToPILImage()
        img = np.array(img)
        # img_2 = img
        # img_2[:, :, 0] = img[:, :, 2]
        # img_2[:, :, 1] = img[:, :, 0]
        # img_2[:, :, 2] = img[:, :, 1]
        #
        # img_2 = Image.fromarray(img_2.astype('uint8')).convert('RGB')
        # img_2.show()
        label_path = save_path + '/' + str(label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        # img.save(label_path +'/{}.jpg'.format(idx), quality=95)
        cv2.imwrite((label_path +'/{}.jpg'.format(idx)), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path,ep.faces_lfw_0.4_112x112,faces_emore",
                        default='iccv', type=str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = '/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1'
    # load_mx_rec('/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1')

    if args.rec_path == 'faces_emore':

        bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
        for i in range(len(bin_files)):
            load_bin(rec_path + '/' + (bin_files[i]+'.bin'), rec_path + '/' + bin_files[i], conf.test_transform)

    elif args.rec_path == 'faces_msceleb_clean_112x112':

        bin_files = ['val_lfw_new']
        for i in range(len(bin_files)):
            load_bin(rec_path + '/' + (bin_files[i]+'.bin'), rec_path + '/' + bin_files[i], conf.test_transform)

    elif args.rec_path == 'iccv':

        # bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
        bin_files = ['agedb_30', 'cfp_fp', 'lfw']

        for i in range(len(bin_files)):
            load_bin(rec_path + '/' + (bin_files[i]+'.bin'), rec_path + '/' + bin_files[i], conf.test_transform)

