
import sys
import os
project_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_path)

import torch
import pickle
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
from model import MobileFaceNet_sor
from model import MobileFaceNet_22
from test_by_zhang.RocCurve import RocCurve

import mxnet as mx

class local_bin_test():
    def __init__(self, batch_size, bin_path, model_path):
        self.bin_file_type = bin_path
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.bins, self.issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
        self.model = MobileFaceNet_22(embedding_size = 512).to(0)
        self.model.load_state_dict(torch.load(model_path))
        self.model_path = model_path
        self.roc = RocCurve()


    def extract_feature(self):
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        embedding_feature = np.zeros([len(self.bins), 512])
        predict_result = np.zeros([len(self.issame_list), 2], dtype=np.float32)

        batch_index = 0
        batch_list = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(self.bins))):
                _bin = self.bins[i]
                img = mx.image.imdecode(_bin).asnumpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = Image.fromarray(img.astype(np.uint8))

                if i < batch_index + self.batch_size:
                    img_tensor = transform(img)
                    batch_list.append(img_tensor)
                else:
                    batch_tensor = torch.stack([elem for elem in batch_list], 0)
                    feautre = self.model(batch_tensor.to(0)).cpu().numpy()
                    embedding_feature[batch_index:batch_index+self.batch_size, :] = feautre
                    #clear batch list
                    batch_list = []
                    img_tensor = transform(img)
                    batch_list.append(img_tensor)
                    batch_index = i

            if len(batch_list) > 0:
                batch_tensor = torch.stack([elem for elem in batch_list], 0)
                feautre = self.model(batch_tensor.to(0)).cpu().numpy()
                embedding_feature[batch_index:] = feautre

        feature1 = embedding_feature[0::2]
        feature2 = embedding_feature[1::2]
        assert feature1.shape[0] == feature2.shape[0]
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        label = np.array(self.issame_list, dtype=int)

        predict_result[:,0] = dist
        predict_result[:,1] = label

        roc_img_path = self.model_path.replace('.pth', '_ROC.png')
        roc_txt_result = self.model_path.replace('.pth', '_ROC.txt')
        roc_info_result = self.model_path.replace('.pth', '_info.txt')
        np.savetxt(roc_txt_result, predict_result,fmt='%.4e' )
        roc_info, info_dict = self.roc.draw_roc(-dist, label, roc_img_path)
        f = open(roc_info_result, 'w')
        for line in roc_info:
            f.write("{}\n".format(line))
        f.close()



if __name__ == "__main__":
    batch_size = 100
    bin_path = "/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1/cfp_fp.bin"
    model_path = "/home/yeluyue/dl/Datasets/work_space/ICCV2019_workspace/Msceleb_clean_zhu_mobilefacenet22_0.4_lr_e-1/save/model_2019-05-14-09-57_accuracy:0.0_step:2123636_None.pth"
    bin_test = local_bin_test(batch_size, bin_path, model_path)
    bin_test.extract_feature()

    # roc_result_path = "/home/zkx/Project/pytorch_models/FaceRecognition/finetune_model/model_2019-05-09-02-37_accuracy_0.941857142857143_step_1035920_final_ROC.txt"
    # roc_img_path = roc_result_path.replace('.txt', '.png')
    # data = np.loadtxt(roc_result_path)
    # roc = RocCurve()
    # roc_info, info_dict = roc.draw_roc(-data[:,0], data[:,1], roc_img_path)
    # for elem in roc_info:
    #     print(elem)





