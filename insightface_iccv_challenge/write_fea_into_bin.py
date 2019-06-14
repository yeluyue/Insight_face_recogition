#write model feature into bin
import os
import torch
import numpy as np
from insightface_iccv_challenge.data_io.data_io_folder import get_test_list_loader
from model import MobileFaceNet_22 ,MobileFaceNet_y2, MobileFaceNet_23, MobileFaceNet_21, MobileFaceNet_sor, MobileFaceNet_11
from model import MobileFaceNet_y2_se, MobileFaceNet_y2_4
from model import MobileFaceNet_sor, MobileFaceNet_22, MobileFaceNet_y2, MobileFaceNet_23, MobileFaceNet_y2_se,mf_y2_mbv2_t6, MobileFaceNet_y2_6
from models.mf_y2_shuffle.mobilefacenet_big import MobileFaceNet_y2_shufflev2g4
from models.mf_y2_shuffle_SE.mobilefacenet_big_SE import MobileFaceNet_y2_shufflev2g4_SE
from models.mobilenetv3.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large, MobileNetV3_Large_ex2

from models.mf_sknet.mf_y2_sknet import mf_y2_sknet,  mf_y2_sknet_res, mf_y2_res_sknet


from tqdm import tqdm
import struct
class IccvTest():
    def __init__(self, model_path, image_root_path, image_list_path, batch_size, workers, dst_path):
        self.loader,self.img_num= get_test_list_loader(image_root_path, image_list_path,
                                                           batch_size, workers)
        self.embedding_size = 512
    #define your network
        self.model = mf_y2_sknet_res(512)
        self.model.load_state_dict(torch.load(model_path))
        bin_name = os.path.basename(model_path).replace(".pth", ".bin")
        self.dst_path = "{}/{}".format(dst_path, bin_name)
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def write_bin(self, m):
        rows, cols = m.shape
        with open(self.dst_path, 'wb') as f:
            f.write(struct.pack('4i', rows, cols, cols * 4, 5))
            f.write(m.data)


    def extract_features(self):
        feature = np.zeros((self.img_num, self.embedding_size), dtype = np.float32)
        index = 0
        self.model.eval()
        with torch.no_grad():
            for batch, prefix in tqdm(iter(self.loader)):
                batch_size = batch.shape[0]
                result = self.model(batch.to(self.device))
                feature[index: index + batch_size, :] = result.cpu().numpy()
                index = index + batch_size
        assert index == self.img_num
        self.write_bin(feature)


if __name__ == '__main__':
    model_path = "/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_zhu_mf_y2_sknet_res_0.4_lr_e-1/models/model_2019-06-12-14-56_accuracy:0.9862105_step:690620_final.pth"
    image_root_path = "/home/yeluyue/yeluyue/ICCV_challenge/test_data/iccv19-challenge-data"
    image_list_path = "/home/yeluyue/yeluyue/ICCV_challenge/test_data/iccv19-challenge-data/filelist.txt"
    batch_size = 800
    #how many process use to load data
    workers = 32
    dst_path = "/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_zhu_mf_y2_sknet_res_0.4_lr_e-1/features"
    test = IccvTest(model_path, image_root_path, image_list_path, batch_size, workers, dst_path)
    test.extract_features()


