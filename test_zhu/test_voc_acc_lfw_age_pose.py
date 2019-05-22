import argparse

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from config import get_config
from torch.autograd import Variable
import torch
import math

from resnet import resnet18,resnet34
from model_lib import FaceNet_20
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm

from veri_roc_acc_lfw_age_pose import test_in_testing


from ShuffleNet_V2.shuffleNet_V2 import ShuffleNetV2
from mobilenet_v2.MobileNetV2 import MobileNetV2




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for test face verification')

    parser.add_argument("--load_mode", help="which trained model to load for testing",
                        # default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/models/model_mobilefacenet.pth',
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_mobilenetv2_0.1_lr_e-2/models/model_2019-04-30-20-13_accuracy:0.8379999999999999_step:51796_None.pth'
                        )
    parser.add_argument("--pairs_path", help="where to load pairs.txt",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112/lfw_new',
                        type=str)
    parser.add_argument("--txt_save", help="where to save the results in .txt",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_shufflenetv2_0.1_lr_e-2/txt',
                        type=str)

    args = parser.parse_args()
    model_path = args.load_mode
    txt_save_path = args.txt_save
    pairs_path = args.pairs_path
    conf = get_config()

    print('/.................................................................................')
    print('loading the test model:', model_path)
    # 加载mobilefacenet
    # model = MobileFaceNet(conf.embedding_size).to(conf.device)

    # 加载resnet
    # model = resnet18().to(conf.device)
    # model = resnet34().to(conf.device)

    # 加载facenet20
    # model = FaceNet_20().to(conf.device)]

    # 加载shuffle_v2
    # model = ShuffleNetV2().to(conf.device)

    # 加载mobilenet_v2
    model = MobileNetV2(n_class=512).to(conf.device)

    model.load_state_dict(torch.load(model_path))
    print('model loaded.')

    test_in_testing(model, txt_save_path, True)