from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
import os

def get_config(training = True):
    conf = edict()
    #conf.data_path = Path('data')
    #conf.work_path = Path('work_space/')

    conf.data_path = os.path.join(os.getcwd(), 'data/')
    conf.rec_data_path = '/home/yeluyue/yeluyue/DataSet/MsCeleb_clean_10k_rec'
    conf.work_path = os.path.join(os.getcwd(), 'work_space/')
    conf.model_path = conf.work_path + 'models'
    conf.log_path = conf.work_path + 'log'
    conf.save_path = conf.work_path+ 'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    #conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.device = torch.device("cuda:0" )
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path + 'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path + 'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path + 'faces_emore'
    conf.lfw_folder = conf.data_path + 'faces_lfw_0.4_112x112'
    # conf.msceleb_folder = '/home/yeluyue/yeluyue/DataSet'
    conf.msceleb_folder = '/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1'
    # conf.msceleb_folder = '/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m_10k'


    conf.batch_size = 100 # irse net depth 50 
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path + 'log'
        conf.save_path = conf.work_path + 'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [5, 10, 16, 24, 30, 35, 40, 45]
        # conf.milestones = [5, 7, 10, 12, 15, 17]
        # conf.milestones = [12, 15, 18]

        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()
        conf.fine_tune = False
        conf.optim = 'sgd'  # 'adabound'
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path + 'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
        conf.fine_tune = True

    return conf