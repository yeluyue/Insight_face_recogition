# -- coding: utf-8 --

""""
用于人脸识别的测试和独立验证
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as trans
from torch.autograd import Variable
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io
import pickle
import mxnet as mx
import cv2
plt.switch_backend('agg')


class Get_feature_label(object):
    def __init__(self, config, model):
        super(Get_feature_label, self).__init__()
        if config.pro == 'test':
            self.save_txt = True
            self.save_txt_path = config.save_txt_path + '/distance_label.txt'
        elif config.pro == 'val':
            self.save_txt = False
        elif config.pro == 'train':
            self.save_txt = False
        else:
            print('the key of processing is not correct')

        self.pairs_path = config.pair_path
        self.file_ext = config.file_ext
        self.dataset_path = config.dataset_path
        self.embedding_size = config.embedding_size
        self.model = model

    def get_distance(self):

        pairs = read_pairs(os.path.join(self.pairs_path, 'pairs.txt'))
        # 建立图像名称与图像地址的字典，并保存图像对是否是同人信息
        name_space_path, imgs_list, self.actual_issame = get_path_name(self.dataset_path, pairs, self.file_ext)
        # 进行图像的特征向量，并进行图像与特征向量的映射
        dict_name_features = get_features_dict(name_space_path, self.model)
        embeddings = np.zeros([len(imgs_list), self.embedding_size])

        for i in range(len(imgs_list)):
            img_name = imgs_list[i]
            embed = dict_name_features[img_name]
            embeddings[i] = embed
        # print('/ 映射特征向量至图像名称完成')

        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        self.distance = np.zeros(embeddings2.shape[0])

        for i in range(embeddings2.shape[0]):
            self.distance[i] = np.linalg.norm(embeddings1[i] - embeddings2[i])

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


def test_in_testing(model, save_txt_path, save_roc_info):
    """
    :param model:加载的模型参数
    :param save_txt_path: 需要将生成的distance_label.txt文件和勾画的roc文件保存的地址
    :param save_roc_info: 是否需要保存roc的具体参数,True or False
    :return:
    """
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(
        '/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1')
    embeddings_agedb_30, embeddings_cpf_fp, embeddings_lfw = get_features(agedb_30, cfp_fp, lfw, model)
    dist_agedb, dist_cfp, dist_lfw, dist_all = get_distances(embeddings_agedb_30, embeddings_cpf_fp, embeddings_lfw)

    ###################################### agedb  ########################################################################
    fpr_agedb, tpr_agedb, thresholds_agedb = roc_curve(agedb_30_issame, -1.0 * dist_agedb, pos_label=1)
    output_figure_path_agedb = save_txt_path + '/agedb_ROC.png'
    roc_info_agedb, info_dict_agedb = draw_test_roc(fpr_agedb, tpr_agedb, thresholds_agedb, output_figure_path_agedb, output_figure_path_agedb)
    # save transform result
    if save_roc_info:
        roc_info_path = output_figure_path_agedb.replace('_ROC.png', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict_agedb[1e-4]
        score_level2 = info_dict_agedb[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.1
        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1,
                                                                                                           score_level2,
                                                                                                           score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info_agedb:
            f.write("{}\n".format(line))
        f.close()
    ###################################### cfp  ########################################################################

    fpr_cfp, tpr_cfp, thresholds_cfp = roc_curve(cfp_fp_issame, -1.0 * dist_cfp, pos_label=1)
    output_figure_path_cfp = save_txt_path + '/cfp_ROC.png'
    roc_info_cfp, info_dict_cfp = draw_test_roc(fpr_cfp, tpr_cfp, thresholds_cfp, output_figure_path_cfp,
                                                    output_figure_path_cfp)
    if save_roc_info:
        roc_info_path = output_figure_path_cfp.replace('_ROC.png', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict_cfp[1e-4]
        score_level2 = info_dict_cfp[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.1

        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1,
                                                                                                           score_level2,
                                                                                                           score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info_cfp:
            f.write("{}\n".format(line))
        f.close()
    ###################################### lfw  ########################################################################

    fpr_lfw, tpr_lfw, thresholds_lfw = roc_curve(lfw_issame, -1.0 * dist_lfw, pos_label=1)
    output_figure_path_lfw = save_txt_path + '/lfw_ROC.png'
    roc_info_lfw, info_dict_lfw = draw_test_roc(fpr_lfw, tpr_lfw, thresholds_lfw, output_figure_path_lfw, output_figure_path_lfw)
    # save transform result
    if save_roc_info:
        roc_info_path = output_figure_path_lfw.replace('_ROC.png', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict_lfw[1e-4]
        score_level2 = info_dict_lfw[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.1
        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1,
                                                                                                           score_level2,
                                                                                                           score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info_lfw:
            f.write("{}\n".format(line))
        f.close()
    ###################################### all of the val datasets #####################################################

    issame_all = np.concatenate((agedb_30_issame, cfp_fp_issame, lfw_issame), 0)
    fpr_all, tpr_all, thresholds_all = roc_curve(issame_all, -1.0 * dist_all, pos_label=1)

    output_figure_path_all = save_txt_path + '/Three_ROC.png'
    roc_info_all, info_dict_all = draw_test_roc(fpr_all, tpr_all, thresholds_all, output_figure_path_all, output_figure_path_all)
    # save transform result
    if save_roc_info:
        roc_info_path = output_figure_path_all.replace('_ROC.png', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict_all[1e-4]
        score_level2 = info_dict_all[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.1

        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1,
                                                                                                           score_level2,
                                                                                                           score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info_all:
            f.write("{}\n".format(line))
        f.close()

def get_distances(embeddings_agedb_30, embeddings_cpf_fp, embeddings_lfw):

    embeddings_agedb_30_1 = embeddings_agedb_30[0::2]
    embeddings_agedb_30_2 = embeddings_agedb_30[1::2]
    diff_agedb = np.subtract(embeddings_agedb_30_1, embeddings_agedb_30_2)
    dist_agedb = np.sum(np.square(diff_agedb), 1)

    embeddings_cpf_fp_1 = embeddings_cpf_fp[0::2]
    embeddings_cpf_fp_2 = embeddings_cpf_fp[1::2]
    diff_cfp = np.subtract(embeddings_cpf_fp_1, embeddings_cpf_fp_2)
    dist_cfp = np.sum(np.square(diff_cfp), 1)

    embeddings_lfw_1 = embeddings_lfw[0::2]
    embeddings_lfw_2 = embeddings_lfw[1::2]
    diff_lfw = np.subtract(embeddings_lfw_1, embeddings_lfw_2)
    dist_lfw = np.sum(np.square(diff_lfw), 1)

    embeddings_1 = np.concatenate((embeddings_agedb_30_1, embeddings_cpf_fp_1, embeddings_lfw_1), 0)
    embeddings_2 = np.concatenate((embeddings_agedb_30_2, embeddings_cpf_fp_2, embeddings_lfw_2), 0)
    diff_all = np.subtract(embeddings_1, embeddings_2)
    dist_all = np.sum(np.square(diff_all), 1)

    return dist_agedb, dist_cfp, dist_lfw, dist_all

def get_val_data(data_path):
    bin_agedb = data_path + '/agedb_30.bin'
    bin_cfp = data_path + '/cfp_fp.bin'
    bin_lfw = data_path + '/lfw.bin'
    transform = trans.Compose([
        # transforms.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    agedb_30, agedb_30_issame = pickle.load(open(bin_agedb, 'rb'), encoding='bytes')
    cfp_fp, cfp_fp_issame = pickle.load(open(bin_cfp, 'rb'), encoding='bytes')
    lfw, lfw_issame = pickle.load(open(bin_lfw, 'rb'), encoding='bytes')

    agedb_30_tensor = torch.zeros(len(agedb_30), 3, 112, 112)
    cfp_fp_tensor = torch.zeros(len(cfp_fp), 3, 112, 112)
    lfw_tensor = torch.zeros(len(lfw), 3, 112, 112)

    for i in range(len(agedb_30)):
        _bin = agedb_30[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img)
        agedb_30_tensor[i, :, :, :] = img_tensor

    for i in range(len(cfp_fp)):
        _bin = cfp_fp[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img)
        cfp_fp_tensor[i, :, :, :] = img_tensor

    for i in range(len(lfw)):
        _bin = lfw[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img)
        lfw_tensor[i, :, :, :] = img_tensor

    return agedb_30_tensor, cfp_fp_tensor, lfw_tensor, agedb_30_issame, cfp_fp_issame, lfw_issame


def get_features(agedb_30, cfp_fp, lfw, model):
    batch_size = 500
    embeddings_agedb_30 = np.zeros([len(agedb_30), 512])
    embeddings_cpf_fp = np.zeros([len(cfp_fp), 512])
    embeddings_lfw = np.zeros([len(lfw), 512])
    idx_agedb_30 = 0
    idx_cpf_fp = 0
    idx_lfw = 0
    model.eval()

    with torch.no_grad():
        while idx_agedb_30 + batch_size <= len(agedb_30):
            batch = agedb_30[idx_agedb_30:idx_agedb_30 + batch_size]
            embeddings_agedb_30[idx_agedb_30:idx_agedb_30 + batch_size] = model(batch.to(0)).cpu().numpy()
            idx_agedb_30 += batch_size
        if idx_agedb_30 < len(agedb_30):
            batch = agedb_30[idx_agedb_30:]
            embeddings_agedb_30[idx_agedb_30:] = model(batch.to(0)).cpu().numpy()

        while idx_cpf_fp + batch_size <= len(cfp_fp):
            batch = cfp_fp[idx_cpf_fp:idx_cpf_fp + batch_size]
            embeddings_cpf_fp[idx_cpf_fp:idx_cpf_fp + batch_size] = model(batch.to(0)).cpu().numpy()
            idx_cpf_fp += batch_size
        if idx_cpf_fp < len(cfp_fp):
            batch = cfp_fp[idx_agedb_30:]
            embeddings_cpf_fp[idx_cpf_fp:] = model(batch.to(0)).cpu().numpy()

        while idx_lfw + batch_size <= len(lfw):
            batch = lfw[idx_lfw:idx_lfw + batch_size]
            embeddings_lfw[idx_lfw:idx_lfw + batch_size] = model(batch.to(0)).cpu().numpy()
            idx_lfw += batch_size
        if idx_lfw < len(lfw):
            batch = lfw[idx_lfw:]
            embeddings_lfw[idx_lfw:] = model(batch.to(0)).cpu().numpy()

    return embeddings_agedb_30, embeddings_cpf_fp, embeddings_lfw



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

