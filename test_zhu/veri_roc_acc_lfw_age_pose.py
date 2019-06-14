# -- coding: utf-8 --

""""
用于人脸识别的测试和独立验证
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as trans
from torch.autograd import Variable
import torch
from easydict import EasyDict as edict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io
plt.switch_backend('agg')



def get_config_voc(testing=False):
    config = edict()
    config.pro = 'train'
    #输入pair.txt所在位置
    #输出图像保存的格式
    config.file_ext = 'png'
    config.lfw_pairs_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112'
    config.lfw_dataset_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112/lfw_new'

    config.calfw_pairs_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112'
    config.calfw_dataset_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112/calfw_imgs'

    config.cplfw_pairs_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112'
    config.cplfw_dataset_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112/cplfw_imgs'

    config.embedding_size = 512

    #--------------------Training Config ------------------------
    if testing:
        config.save_txt_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/test_txt/test'
        config.pro = 'test'

    return config

class Get_feature_label(object):
    def __init__(self, config, model):
        super(Get_feature_label, self).__init__()
        if config.pro == 'test':
            self.save_txt = True
            self.save_txt_path = config.save_txt_path + '/distance_label_lfw_age_pose.txt'
        elif config.pro == 'val':
            self.save_txt = False
        elif config.pro == 'train':
            self.save_txt = False
        else:
            print('the key of processing is not correct')

        self.file_ext = config.file_ext

        self.lfw_pairs_path = config.lfw_pairs_path
        self.lfw_dataset_path = config.lfw_dataset_path

        self.calfw_pairs_path = config.calfw_pairs_path
        self.calfw_dataset_path = config.calfw_dataset_path

        self.cplfw_pairs_path = config.cplfw_pairs_path
        self.cplfw_dataset_path = config.cplfw_dataset_path

        self.embedding_size = config.embedding_size
        self.model = model

    def get_distance(self):

        #.....................................计算lfw验证集内的distance和label............................................
        lfw_pairs = read_pairs(os.path.join(self.lfw_pairs_path, 'pairs.txt'))
        # 建立图像名称与图像地址的字典，并保存图像对是否是同人信息
        lfw_name_space_path, lfw_imgs_list, self.lfw_actual_issame = get_path_name(self.lfw_dataset_path, lfw_pairs, self.file_ext)
        # 进行图像的特征向量，并进行图像与特征向量的映射
        lfw_dict_name_features = get_features_dict(lfw_name_space_path, self.model)
        lfw_embeddings = np.zeros([len(lfw_imgs_list), self.embedding_size])

        for i in range(len(lfw_imgs_list)):
            lfw_img_name = lfw_imgs_list[i]
            embed = lfw_dict_name_features[lfw_img_name]
            lfw_embeddings[i] = embed
        # print('/ 映射特征向量至图像名称完成')

        lfw_embeddings1 = lfw_embeddings[0::2]
        lfw_embeddings2 = lfw_embeddings[1::2]
        self.lfw_distance = np.zeros(lfw_embeddings2.shape[0])

        for i in range(lfw_embeddings2.shape[0]):
            self.lfw_distance[i] = np.linalg.norm(lfw_embeddings1[i] - lfw_embeddings2[i])

        #.....................................计算calfw验证集内的distance和label..........................................

        calfw_pairs = read_calfwpairs(os.path.join(self.calfw_pairs_path, 'pairs_CALFW.txt'))
        # 建立图像名称与图像地址的字典，并保存图像对是否是同人信息
        calfw_name_space_path, calfw_imgs_list, self.calfw_actual_issame = get_path_calfwname(self.calfw_dataset_path, calfw_pairs,
                                                                                   self.file_ext)
        # 进行图像的特征向量，并进行图像与特征向量的映射
        calfw_dict_name_features = get_features_dict(calfw_name_space_path, self.model)
        calfw_embeddings = np.zeros([len(calfw_imgs_list), self.embedding_size])

        for i in range(len(calfw_imgs_list)):
            calfw_img_name = calfw_imgs_list[i]
            embed = calfw_dict_name_features[calfw_img_name]
            calfw_embeddings[i] = embed
        # print('/ 映射特征向量至图像名称完成')

        calfw_embeddings1 = calfw_embeddings[0::2]
        calfw_embeddings2 = calfw_embeddings[1::2]
        self.calfw_distance = np.zeros(calfw_embeddings2.shape[0])

        for i in range(calfw_embeddings2.shape[0]):
            self.calfw_distance[i] = np.linalg.norm(calfw_embeddings1[i] - calfw_embeddings2[i])

        # .....................................计算cplfw验证集内的distance和label..........................................

        cplfw_pairs = read_calfwpairs(os.path.join(self.cplfw_pairs_path, 'pairs_CPLFW.txt'))
        # 建立图像名称与图像地址的字典，并保存图像对是否是同人信息
        cplfw_name_space_path, cplfw_imgs_list, self.cplfw_actual_issame = get_path_cplfwname(
            self.cplfw_dataset_path, cplfw_pairs,
            self.file_ext)
        # 进行图像的特征向量，并进行图像与特征向量的映射
        cplfw_dict_name_features = get_features_dict(cplfw_name_space_path, self.model)
        cplfw_embeddings = np.zeros([len(cplfw_imgs_list), self.embedding_size])

        for i in range(len(cplfw_imgs_list)):
            cplfw_img_name = cplfw_imgs_list[i]
            embed = cplfw_dict_name_features[cplfw_img_name]
            cplfw_embeddings[i] = embed
        # print('/ 映射特征向量至图像名称完成')

        cplfw_embeddings1 = cplfw_embeddings[0::2]
        cplfw_embeddings2 = cplfw_embeddings[1::2]
        self.cplfw_distance = np.zeros(cplfw_embeddings2.shape[0])

        for i in range(cplfw_embeddings2.shape[0]):
            self.cplfw_distance[i] = np.linalg.norm(cplfw_embeddings1[i] - cplfw_embeddings2[i])

        self.distance = np.concatenate((self.lfw_distance, self.calfw_distance, self.cplfw_distance), 0)
        self.actual_issame = np.concatenate((self.lfw_actual_issame, self.calfw_actual_issame, self.cplfw_actual_issame), axis=0)

        return self.distance, self.actual_issame

    def get_distance_label_txt(self):
        if self.save_txt:
            print('/start writing the distance.....................')
            with open(self.save_txt_path, 'a+') as fp:
                for i in range(len(self.distance)):
                    dist = self.distance[i]
                    fp.write('%-8f' % dist)
                    fp.write('%4d' % self.actual_issame[i])
                    fp.write("\n")

            fp.close()

def test_in_training(model):
    '''

    :param model:
    :return:
    程序流程：获取distance和label，再计算roc参数
    '''

    config = get_config_voc()
    Get_distance = Get_feature_label(config, model)

    distance, actual_issame = Get_distance.get_distance()
    fpr, tpr, thresholds = roc_curve(actual_issame, -1.0 * distance, pos_label=1)

    fpr_7 = np.around(fpr, decimals=7)
    index = np.argmin(abs(fpr_7 - 10 ** -3))
    # same threshold match the multi-accuracy
    index_all = np.where(fpr_7 == fpr_7[index])
    # select max accuracy
    max_acc = np.max(tpr[index_all])
    best_thresholds = np.max(abs(thresholds[index_all]))
    # print index
    accuracy = np.around(max_acc, decimals=7)

    return tpr, fpr, accuracy, best_thresholds, thresholds



def test_in_testing(model, save_txt_path,save_roc_info):
    """

    :param model:加载的模型参数
    :param save_txt_path: 需要将生成的distance_label.txt文件和勾画的roc文件保存的地址
    :param save_roc_info: 是否需要保存roc的具体参数,True or False
    :return:
    """
    config = get_config_voc(testing=True)
    config.save_txt_path = save_txt_path

    Get_distance = Get_feature_label(config, model)
    distance, actual_issame = Get_distance.get_distance()
    Get_distance.get_distance_label_txt()

    fpr, tpr, thresholds = roc_curve(actual_issame, -distance, pos_label=1)

    output_figure_path = save_txt_path + '/distace_label_lfw_age_pose_ROC.png'
    roc_info, info_dict = draw_test_roc(fpr, tpr, thresholds, output_figure_path, output_figure_path)
    # save transform result
    if save_roc_info:
        roc_info_path = output_figure_path.replace('_ROC.png', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict[1e-4]
        score_level2 = info_dict[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.1

        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1,
                                                                                                           score_level2,
                                                                                                           score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info:
            f.write("{}\n".format(line))
        f.close()

    fpr_7 = np.around(fpr, decimals=7)
    index = np.argmin(abs(fpr_7 - 10 ** -3))
    # same threshold match the multi-accuracy
    index_all = np.where(fpr_7 == fpr_7[index])
    # select max accuracy
    max_acc = np.max(tpr[index_all])
    best_thresholds = np.max(abs(thresholds[index_all]))
    # print index
    accuracy = np.around(max_acc, decimals=7)

    print('accuracy, best_thresholds: ', accuracy, best_thresholds)




def draw_test_roc(fpr, tpr, thresholds, output_figure_path, title=None):
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    # save the fpr acc threshold
    roc_info = []
    info_dict = {}  # [tolerance] :  threshold
    for tolerance in [10 ** -7, 10 ** -6, 5.0 * 10 ** -6, 10 ** -5, 10 ** -4, 1.2 * 10 ** -4, 10 ** -3, 10 ** -2,
                      10 ** -1]:
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        # same threshold match the multi-accuracy
        index_all = np.where(fpr == fpr[index])
        # select max accuracy
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))
        # print index
        x, y = fpr[index], max_acc

        plt.plot(x, y, 'x')
        plt.text(x, y, "({:.7f}, {:.7f}) threshold={:.7f}".format(x, y, threshold))
        temp_info = 'fpr\t{}\tacc\t{}\tthreshold\t{}'.format(tolerance, round(max_acc, 5), threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = threshold

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {}'.format(title))
    # plt.legend(loc="lower right")
    fig.savefig(output_figure_path)
    plt.close(fig)
    return roc_info, info_dict

def draw_train_roc(fpr, tpr, thresholds, title=None):
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    # save the fpr acc threshold
    roc_info = []
    info_dict = {}  # [tolerance] :  threshold
    for tolerance in [10 ** -7, 10 ** -6, 5.0 * 10 ** -6, 10 ** -5, 10 ** -4, 1.2 * 10 ** -4, 10 ** -3, 10 ** -2,
                      10 ** -1]:
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        # same threshold match the multi-accuracy
        index_all = np.where(fpr == fpr[index])
        # select max accuracy
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))
        # print index
        x, y = fpr[index], max_acc

        plt.plot(x, y, 'x')
        plt.text(x, y, "({:.7f}, {:.7f}) threshold={:.7f}".format(x, y, threshold))
        temp_info = 'fpr\t{}\tacc\t{}\tthreshold\t{}'.format(tolerance, round(max_acc, 5), threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = threshold

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {}'.format(title))
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

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

def read_calfwpairs(pairs_filename):
    """
    按行读取pairs.txt文件，并按照行序返回一个列表
    :param pairs_filename:
    :return: 图像名称组成的list
    """
    pairs = []
    count = 0
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[:]:
            pair = line.strip().split()
            pairs.append(pair)
            count += 1
    return pairs

def get_path_name(dataset_path, pairs, file_ext):
    '''
    :param dataset_path: pairs.txt文件内原始图像保存的地址
    :param pairs: 按行读取pair.txt返回的列表
    :param file_ext:原始图像保存的格式
    :return name_space_path --返回一个字典，保存的是类似于{图像名称：[图像地址]}，没有重复元素
    :return imgs_path  pairs.txt内图像包含的图像对应地址
    :return imgs_list 按照pair.txt的行顺序保存图像名称（包含重复图像名称）
    :return actual_issame 按照pair.txt的行顺序按行保存图像名称（包含重复图像名称）
    '''

    name_space_path = {}
    imgs_list = []
    actual_issame = []

    for pair in pairs:
        # img0_name,img1_name分别指的是同一行内两张图像的名字
        if len(pair) == 3:
            img0_name = pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext
            img1_name = pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext
            issame = 1

            imgs_list.append(img0_name)
            imgs_list.append(img1_name)
            actual_issame.append(issame)

            img0_path = os.path.join(dataset_path, pair[0], img0_name)
            img1_path = os.path.join(dataset_path, pair[0], img1_name)

            name_space_path[img0_name] = []
            name_space_path[img1_name] = []
            name_space_path.setdefault(img0_name).append(img0_path)
            name_space_path.setdefault(img1_name).append(img1_path)


        elif len(pair) == 4:
            img0_name = pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext
            img1_name = pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext
            issame = 0

            imgs_list.append(img0_name)
            imgs_list.append(img1_name)
            actual_issame.append(issame)

            img0_path = os.path.join(dataset_path, pair[0], img0_name)
            img1_path = os.path.join(dataset_path, pair[2], img1_name)

            name_space_path[img0_name] = []
            name_space_path[img1_name] = []
            name_space_path.setdefault(img0_name).append(img0_path)
            name_space_path.setdefault(img1_name).append(img1_path)

    # print('the  number of total imgs which exist in pairs.txt:', len(name_space_path))
    return name_space_path, imgs_list, actual_issame


def get_path_calfwname(dataset_path, pairs, file_ext):


    name_space_path = {}
    imgs_list = []
    actual_issame = []

    for i in range(int(len(pairs) / 4)):

        # img0_name,img1_name分别指的是同一行内两张图像的名字
        pair1 = pairs[int(2*i)]
        pair2 = pairs[int(2*i + 1)]
        img0_name_jpg = pair1[0]
        img1_name_jpg = pair2[0]
        img0_name = img0_name_jpg.replace('jpg', 'png')
        img1_name = img1_name_jpg.replace('jpg', 'png')

        people0_name = img0_name[0:(len(img0_name)-9)]
        issame = 1

        imgs_list.append(img0_name)
        imgs_list.append(img1_name)
        actual_issame.append(issame)

        img0_path = os.path.join(dataset_path, people0_name, img0_name)
        img1_path = os.path.join(dataset_path, people0_name, img1_name)

        name_space_path[img0_name] = []
        name_space_path[img1_name] = []
        name_space_path.setdefault(img0_name).append(img0_path)
        name_space_path.setdefault(img1_name).append(img1_path)

    for i in range(int(len(pairs)/4)):

        s = len(pairs)/2
        pair1 = pairs[int(s + 2*i)]
        pair2 = pairs[int(s + 2 * i + 1)]
        img0_name_jpg = pair1[0]
        img1_name_jpg = pair2[0]
        img0_name = img0_name_jpg.replace('jpg', 'png')
        img1_name = img1_name_jpg.replace('jpg', 'png')

        people0_name = img0_name[0:(len(img0_name) - 9)]
        people1_name = img1_name[0:(len(img1_name) - 9)]

        issame = 0

        imgs_list.append(img0_name)
        imgs_list.append(img1_name)
        actual_issame.append(issame)

        img0_path = os.path.join(dataset_path, people0_name, img0_name)
        img1_path = os.path.join(dataset_path, people1_name, img1_name)

        name_space_path[img0_name] = []
        name_space_path[img1_name] = []
        name_space_path.setdefault(img0_name).append(img0_path)
        name_space_path.setdefault(img1_name).append(img1_path)

    # print('the  number of total imgs which exist in pairs.txt:', len(name_space_path))
    return name_space_path, imgs_list, actual_issame


def get_path_cplfwname(dataset_path, pairs, file_ext):
    name_space_path = {}
    imgs_list = []
    actual_issame = []

    for i in range(int(3938 / 2)):
        # img0_name,img1_name分别指的是同一行内两张图像的名字
        pair1 = pairs[int(2 * i)]
        pair2 = pairs[int(2 * i + 1)]
        img0_name_jpg = pair1[0]
        img1_name_jpg = pair2[0]
        img0_name = img0_name_jpg.replace('jpg', 'png')
        img1_name = img1_name_jpg.replace('jpg', 'png')

        people0_name = img0_name[0:(len(img0_name) - 9)]
        issame = 1

        imgs_list.append(img0_name)
        imgs_list.append(img1_name)
        actual_issame.append(issame)

        img0_path = os.path.join(dataset_path, people0_name, img0_name)
        img1_path = os.path.join(dataset_path, people0_name, img1_name)

        name_space_path[img0_name] = []
        name_space_path[img1_name] = []
        name_space_path.setdefault(img0_name).append(img0_path)
        name_space_path.setdefault(img1_name).append(img1_path)

    for i in range(2006):
        s = 3938
        pair1 = pairs[int(s + 2 * i)]
        pair2 = pairs[int(s + 2 * i + 1)]
        img0_name_jpg = pair1[0]
        img1_name_jpg = pair2[0]
        img0_name = img0_name_jpg.replace('jpg', 'png')
        img1_name = img1_name_jpg.replace('jpg', 'png')

        people0_name = img0_name[0:(len(img0_name) - 9)]
        people1_name = img1_name[0:(len(img1_name) - 9)]

        issame = 0

        imgs_list.append(img0_name)
        imgs_list.append(img1_name)
        actual_issame.append(issame)

        img0_path = os.path.join(dataset_path, people0_name, img0_name)
        img1_path = os.path.join(dataset_path, people1_name, img1_name)

        name_space_path[img0_name] = []
        name_space_path[img1_name] = []
        name_space_path.setdefault(img0_name).append(img0_path)
        name_space_path.setdefault(img1_name).append(img1_path)

    # print('the  number of total imgs which exist in pairs.txt:', len(name_space_path))
    return name_space_path, imgs_list, actual_issame

def get_features_dict(name_space_path, model):
    model.eval()
    dict_name_features = {}
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # print('/start creating the feature_dictionary.............................')
    with torch.no_grad():
        for key in name_space_path:
            dict_name_features[key] = np.zeros((1, 512))
            for name_path in name_space_path[key]:
                img = Image.open(name_path).convert('RGB')
                #进行数据预处理
                img = test_transform(img)
                #img = torch.tensor(img)
                img = img.unsqueeze_(0)
                img = Variable(img).cuda()
                # img = torch.cat([img,img], dim=0)
                features = model(img)
                features = features.data.cpu().numpy()
                dict_name_features[key] = features
    # print('/the feature_dictionary created .............................')
    return dict_name_features

