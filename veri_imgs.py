# -- coding: utf-8 --

import argparse
import os
from data.data_pipe import de_preprocess, get_train_loader, get_val_data,get_val_lfw_data
from model import Backbone, Arcface, Am_softmax, l2_norm, MobileFaceNet
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import bcolz
from config import get_config
from torch.autograd import Variable
import torch
import math
from resnet import resnet18


def read_pairs(pairs_filename):
    """
    按行读取pairs.txt文件，并按照行序返回一个列表
    :param pairs_filename:
    :return: 图像名称组成的list
    """
    pairs = []
    count = 0
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
            count += 1
    return pairs

def get_features_dict(name_space_path, model):
    model.eval()
    dict_name_features = {}
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print('/start creating the feature_dictionary.............................')
    with torch.no_grad():
        for key in name_space_path:
            dict_name_features[key] = torch.zeros(1, 512)
            for name_path in name_space_path[key]:
                img = Image.open(name_path).convert('RGB')
                #进行数据预处理
                img = test_transform(img)
                #img = torch.tensor(img)
                img = img.unsqueeze_(0)
                img = Variable(img).cuda()
                # img = torch.cat([img,img], dim=0)
                features = model(img)
                dict_name_features[key] = features[0]
    print('/the feature_dictionary created .............................')
    return dict_name_features


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for test face verification')
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='mobilefacenet', type=str)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat, lfw]",
                        default='msceleb', type=str)
    parser.add_argument("--load_mode", help="which trained model to load for testing",
                        # default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/models/model_mobilefacenet.pth',
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/save/model_2019-04-01-23-51_accuracy:0.0_step:106820_final.pth',
                        type=str)
    parser.add_argument("--pairs_path", help="where to load pairs.txt",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112/lfw_new', type=str)
    parser.add_argument("--txt_save", help="where to save the results in .txt",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/test_txt/distance_label_10k_resnet_0.4.txt', type=str)


    args = parser.parse_args()
    model_path = args.load_mode
    txt_path = args.txt_save
    pairs_path = args.pairs_path
    conf = get_config()

    print('/.................................................................................')
    print('loading the test model:', model_path)
    #加载mobilefacenet
    # model = MobileFaceNet(conf.embedding_size).to(conf.device)

    # 加载resnet
    model = resnet18().to(conf.device)

    model.load_state_dict(torch.load(model_path))
    print('model loaded.')
    file_ext = 'png'

    # reading the pairs for image selecting
    pairs = read_pairs(os.path.join(pairs_path, 'pairs.txt'))
    # get the name_space for dictionary
    name_space_path = {}
    imgs_path = []
    imgs_list = []
    actual_issame = []

    for pair in pairs:
        if len(pair) == 3:
            name0 = pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext
            name1 = pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext
            issame = 1

            imgs_list.append(name0)
            imgs_list.append(name1)
            actual_issame.append(issame)

            path0 = os.path.join(pairs_path, pair[0], name0)
            path1 = os.path.join(pairs_path, pair[0], name1)
            name_space_path[name0] = []
            name_space_path[name1] = []

            name_space_path.setdefault(name0).append(path0)
            name_space_path.setdefault(name1).append(path1)
            imgs_path.append(path0)
            imgs_path.append(path1)

        elif len(pair) == 4:
            name0 = pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext
            name1 = pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext
            issame = 0

            imgs_list.append(name0)
            imgs_list.append(name1)
            actual_issame.append(issame)

            path0 = os.path.join(pairs_path, pair[0], name0)
            path1 = os.path.join(pairs_path, pair[2], name1)

            name_space_path[name0] = []
            name_space_path.setdefault(name0).append(path0)

            name_space_path[name1] = []
            name_space_path.setdefault(name1).append(path1)

            imgs_path.append(path0)
            imgs_path.append(path1)

    print(len(imgs_path))

    imgs_path = np.unique(imgs_path)

    print('the  number of total people which exist in pairs.txt:', len(name_space_path))
    print('the  number of total img contained in pairs.txt:', len(imgs_path))

    dict_name_feartures = {}

    dict_name_feartures = get_features_dict(name_space_path, model)
    embeddings = torch.zeros([len(imgs_list), conf.embedding_size])
    embeddings = embeddings.cuda()

    for i in range(len(imgs_list)):
        img_name = imgs_list[i]
        embed = dict_name_feartures[img_name]
        embeddings[i] = embed
    print('/ 映射特征向量至图像名称完成')

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    emdedd = embeddings1 - embeddings2


    embeddings1 = embeddings1.data.cpu().numpy()
    embeddings2 = embeddings2.data.cpu().numpy()

    dist = 0.0
    threshold = 0
    predict_issame = actual_issame

    print('/start writing the distance.....................')

    with open(txt_path, 'a+') as fp:

        for i in range(len(embeddings1)):


            dist = np.linalg.norm(embeddings1[i] - embeddings2[i])
            # n = 2*i + 1
            # m = 2*i
            # img_name0 = imgs_list[m]
            # img_name1 = imgs_list[n]
            fp.write('%-8f' % dist)
            fp.write('%4d' % actual_issame[i])
            fp.write("\n")
            # print(img_name0,img_name1,dist,actual_issame[i])

    fp.close()

    # tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    # fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    #
    # tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    # fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    # acc = float(tp + tn) / len(embeddings1)
    #
    # print('acc', acc)


