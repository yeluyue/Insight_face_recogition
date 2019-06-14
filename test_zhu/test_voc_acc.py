import argparse

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from config import get_config
from torch.autograd import Variable
import torch
import math
# from resnet import resnet18,resnet34
# from model_lib import FaceNet_20
from test_zhu.veri_roc_acc_zhu import test_in_testing
from model import Backbone, Arcface,  Am_softmax, l2_norm
from model import MobileFaceNet_21, MobileFaceNet_22,MobileFaceNet_11,MobileFaceNet_sor, MobileFaceNet_y2, MobileFaceNet_23
from model import MobileFaceNet_y2_se, MobileFaceNet_y2_3, MobileFaceNet_y2_4
from collections import OrderedDict
# from ShuffleNet_V2.shuffleNet_V2 import ShuffleNetV2
# from mobilenet_v2.MobileNetV2 import MobileNetV2
from models.mobilenetv3.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for test face verification')

    parser.add_argument("--load_mode", help="which trained model to load for testing",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_zhu_mobilenet_v3_small_0.4_lr_e-1_finetune/save/model_2019-06-04-12-23_accuracy:0.4726316_step:117800_final.pth'
                        )

    parser.add_argument("--txt_save", help="where to save the results in .txt",
                        default='/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_zhu_mobilenet_v3_small_0.4_lr_e-1_finetune/roc',
                        type=str)

    args = parser.parse_args()
    model_path = args.load_mode
    txt_save_path = args.txt_save
    conf = get_config()

    print('/.................................................................................')
    print('loading the test model:', model_path)
    # 加载mobilefacenet
    model = MobileNetV3_Small(conf.embedding_size).to(conf.device)

    # 加载resnet
    # model = resnet18().to(conf.device)
    # model = resnet34().to(conf.device)

    # 加载facenet20
    # model = FaceNet_20().to(conf.device)]

    # 加载shuffle_v2
    # model = ShuffleNetV2().to(conf.device)

    # 加载mobilenet_v2
    # model = MobileNetV2(n_class=512).to(conf.device)
    # model.load_state_dict(torch.load(model_path))

    model_dict = torch.load(model_path, map_location= torch.device("cuda:0" ))
    keys = iter(model_dict)
    first_layer_name = keys.__next__()
    if first_layer_name[:7].find('module.') >= 0:
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            name_key = k[7:]  # remove `module.`
            new_state_dict[name_key] = v
    # load params
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_dict)

    print('model loaded.')

    test_in_testing(model, txt_save_path, True)