from data.data_pipe import de_preprocess, get_train_loader, get_train_loader_rec, get_val_data, get_val_lfw_data
from model import Backbone, Arcface,  Am_softmax, l2_norm
from models.ShuffleNet_V2.shuffleNet_V2 import ShuffleNetV2
from models.mobilenet_v2.MobileNetV2 import MobileNetV2
from models.squeezenet_v2.squeezenet_v2 import squeezenet1_1
from model import MobileFaceNet_sor, MobileFaceNet_22, MobileFaceNet_y2, MobileFaceNet_23, MobileFaceNet_y2_se

from model import MobileFaceNet_y2_3, MobileFaceNet_y2_4, MobileFaceNet_y2_5, mf_y2_mbv2_t6
# from model.facenet.model_lib import FaceNet_20
from models.mobilenetv3.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from models.mf_sknet.mf_y2_sknet import mf_y2_sknet, mf_y2_sknet_res
from verifacation_zhu import evaluate, evaluate_all
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
from collections import OrderedDict

import bcolz
import os
from models.resnet.resnet import resnet18,resnet34
from adabound import AdaBound
from optimizer import get_parser, create_optimizer
plt.switch_backend('agg')
torch.cuda.set_device(0)


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        print('/......................................................................................................')
        if conf.use_mobilfacenet:
            # mobilefacenet模型初始化
            self.model = mf_y2_sknet_res(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            if conf.net_mode == 'ir' or conf.net_mode == 'ir_se':
                self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
                print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
            elif conf.net_mode == 'resnet18':
                self.model = resnet18().to(conf.device)
                print('resnet18 model generated')
            elif conf.net_mode == 'facenet20':
                self.model = FaceNet_20().to(conf.device)
                print('FaceNet_20 model generated')
            elif conf.net_mode == 'shufflenet_v2':
                # self.model = Network(conf.embedding_size, 1.0).to(conf.device)
                self.model = ShuffleNetV2().to(conf.device)
                print(self.model)
                print('shufflenet_v2 generated')
            elif conf.net_mode == 'mobilenet_v2':
                self.model = MobileNetV2(n_class=512).to(conf.device)
                print('mobilenet_v2:', self.model)
                print('mobilenet_v2 generated')
            elif conf.net_mode == 'squeezenet_v2':
                self.model = squeezenet1_1().to(conf.device)
                print('squeezenet_v2:', self.model)
                print('squeezenet_v2 generated')
            else:
                print('出现错误，请停止训练')

        conf.fine_tune = True
        if conf.fine_tune:
            model_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/Msceleb_clean_zhu_mf_y2_sknet_res_0.4_lr_e-1/save/model_2019-06-12-14-56_accuracy:0.9862105_step:690620_final.pth'
            checkpoint = torch.load(model_path)
            model_dict = torch.load(model_path)
            # model_dict = checkpoint['state_dict']
            keys = iter(model_dict)
            first_layer_name = keys.__next__()
            if first_layer_name[:7].find('module.') >= 0:
                new_state_dict = OrderedDict()
                for k, v in model_dict.items():
                    name_key = k[7:]  # remove `module.`
                    new_state_dict[name_key] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(model_dict)

            # self.model.load_state_dict(checkpoint['state_dict'])
            print("fine_tune the model: ", model_path)

        if not inference:
            self.milestones = conf.milestones
            # 进行训练和测试数据的预处理及相关读取
            self.loader, self.class_num = get_train_loader(conf)

            # 添加自己的验证数据集(同样需要制作.bin和_list.npy文件)
            self.lfw, self.lfw_issame = get_val_lfw_data('/home/yeluyue/yeluyue/InsightFace_Pytorch-master/data/faces_msceleb_clean_112x112')

            self.writer = SummaryWriter(str(conf.log_path))
            self.step = 0
            #初始化arcface模型
            head_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/pre_head/head/all/head_2019-05-20-08-04_accuracy:0.0_step:1139512_None.pth'
            # head_path = '/home/yeluyue/yeluyue/InsightFace_Pytorch-master/work_space/pre_head/head/10k/head_2019-05-17-05-02_accuracy:0.0_step:111910_None.pth'

            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            self.head.load_state_dict(torch.load(head_path))

            print('two model heads generated')

            #将模型分离成是/否包括BN层两部分,用于考虑优化参数的设置

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

###########################################  setting optimization #############################################################

            args = get_parser()

            if conf.optim == 'sgd':
                if conf.use_mobilfacenet:
                    self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                        {'params': paras_only_bn}
                                    ], lr=conf.lr, momentum=conf.momentum)

                elif conf.net_mode == 'ir' or conf.net_mode == 'ir_se' or conf.net_mode == 'squeezenet_v2':
                    self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                        {'params': paras_only_bn}
                                    ], lr=conf.lr, momentum=conf.momentum)

                elif conf.net_mode == 'resnet18' or conf.net_mode == 'shufflenet_v2' or conf.net_mode == 'mobilenet_v2' :
                    self.optimizer = optim.SGD([
                        {'params': self.model.parameters(), 'weight_decay': 4e-5},
                        {'params':  [self.head.kernel], 'weight_decay': 4e-4},
                    ], lr=conf.lr, momentum=conf.momentum)

                elif conf.net_mode == 'facenet20' :
                    self.optimizer = optim.SGD([
                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                        {'params': paras_only_bn}
                    ], lr=conf.lr, momentum=conf.momentum)

            elif conf.optim == 'adabound':
                if conf.use_mobilfacenet:
                    self.optimizer = create_optimizer(args, [
                                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                        {'params': paras_only_bn}
                                    ])
                elif conf.net_mode == 'ir' or conf.net_mode == 'ir_se':
                    self.optimizer = create_optimizer(args, [
                                        {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                        {'params': paras_only_bn}
                                    ])

                elif conf.net_mode == 'resnet18':
                    self.optimizer = create_optimizer(args, [
                        {'params': self.model.parameters(), 'weight_decay': 4e-5},
                        {'params':  [self.head.kernel], 'weight_decay': 4e-4},
                    ])

                elif conf.net_mode == 'facenet20':
                    self.optimizer = create_optimizer(args, [
                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                        {'params': paras_only_bn}
                    ])


            print('训练过程的优化参数设置:', self.optimizer)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//1000
            self.evaluate_every = len(self.loader)//5
            self.save_every = len(self.loader)//1

            print('处理的数据集大小:', len(self.loader))
            print('board_loss_every:', self.board_loss_every)
            print('evaluate_every: ', self.evaluate_every)
            print('save_every: ', self.save_every)
            print(
                '/......................................................................................................')

            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data('/home/yeluyue/yeluyue/ICCV_challenge/train_data/ms1m-retinaface-t1')

        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        #保存训练的mobileface模型,head是Arcface模型参数文件,优化器所处位置文件
        torch.save(
            self.model.state_dict(), save_path + '/' +
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path + '/' +
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path + '/' +
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        #加载断点
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        self.model.load_state_dict(torch.load(save_path + '/{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path + '/head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path + '/optimizer_{}'.format(fixed_str)))
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        # 在验证集上输出训练模型的平均准确度
        # tta代表?
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds, thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr, thresholds)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy, best_thresholds, roc_curve_tensor



    def evaluate_all(self, conf, carray_agedb_30, issame_agedb_30, carray_cpf_ff, issame_cpf_ff,
                     carray_lfw, issame_lfw, nrof_folds=5, tta=False):
        # 在验证集上输出训练模型的平均准确度
        # tta代表?
        self.model.eval()
        idx_agedb_30 = 0
        idx_cpf_ff = 0
        idx_lfw = 0

        embeddings_agedb_30 = np.zeros([len(carray_agedb_30), conf.embedding_size])
        embeddings_cpf_ff = np.zeros([len(carray_cpf_ff), conf.embedding_size])
        embeddings_lfw = np.zeros([len(carray_lfw), conf.embedding_size])
        with torch.no_grad():
            while idx_agedb_30 + conf.batch_size <= len(carray_agedb_30):
                batch = torch.tensor(carray_agedb_30[idx_agedb_30:idx_agedb_30 + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_agedb_30[idx_agedb_30:idx_agedb_30 + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings_agedb_30[idx_agedb_30:idx_agedb_30 + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx_agedb_30 += conf.batch_size
            if idx_agedb_30 < len(carray_agedb_30):
                batch = torch.tensor(carray_agedb_30[idx_agedb_30:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_agedb_30[idx_agedb_30:] = l2_norm(emb_batch)
                else:
                    embeddings_agedb_30[idx_agedb_30:] = self.model(batch.to(conf.device)).cpu()

            #.........................................calculate the set of cpf_ff的值..................................
            while idx_cpf_ff+ conf.batch_size <= len(carray_cpf_ff):
                batch = torch.tensor(carray_cpf_ff[idx_cpf_ff:idx_cpf_ff + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_cpf_ff[idx_cpf_ff:idx_cpf_ff + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings_cpf_ff[idx_cpf_ff:idx_cpf_ff + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx_cpf_ff += conf.batch_size
            if idx_cpf_ff < len(carray_cpf_ff):
                batch = torch.tensor(carray_cpf_ff[idx_cpf_ff:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_cpf_ff[idx_cpf_ff:] = l2_norm(emb_batch)
                else:
                    embeddings_cpf_ff[idx_cpf_ff:] = self.model(batch.to(conf.device)).cpu()

            #.........................................calculate the set of cpf_ff的值..................................
            while idx_lfw + conf.batch_size <= len(carray_lfw):
                batch = torch.tensor(carray_lfw[idx_lfw:idx_lfw + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_lfw[idx_lfw:idx_lfw + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings_lfw[idx_lfw:idx_lfw + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx_lfw += conf.batch_size
            if idx_lfw < len(carray_lfw):
                batch = torch.tensor(carray_lfw[idx_lfw:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings_lfw[idx_lfw:] = l2_norm(emb_batch)
                else:
                    embeddings_lfw[idx_lfw:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds, thresholds = evaluate_all(embeddings_agedb_30, issame_agedb_30,
                                                           embeddings_cpf_ff, issame_cpf_ff,
                                                           embeddings_lfw, issame_lfw, nrof_folds)
        # tpr, fpr, accuracy, best_thresholds, thresholds = test_in_training(self.model)
        # buf = gen_plot(fpr, tpr)
        buf = gen_plot(fpr, tpr, thresholds)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy, best_thresholds, roc_curve_tensor
    
    def find_lr(self,conf,init_value=1e-8,final_value=10.,beta=0.98,bloding_scale=3.,num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            if e == self.milestones[3]:
                self.schedule_lr()
            # if e == self.milestones[4]:
            #     self.schedule_lr()
            # if e == self.milestones[5]:
            #     self.schedule_lr()
            # if e == self.milestones[6]:
            #     self.schedule_lr()
            # if e == self.milestones[7]:
            #     self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                # mobileface处理图像,并输出1*512的特征向量
                embeddings = self.model(imgs)
                # 使用Arcface处理特征向量
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    # 保存训练过程的学习率
                    self.writer.add_scalar('train_lr',  math.log10(self.optimizer.param_groups[1]['lr']), self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    # 在验证集上计算准确率和roc曲线
                    # 叶路月写的验证代码
                    # accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)

                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                               self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    print('agedb_30: accuracy ', accuracy)

                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    print('lfw: accuracy ', accuracy)

                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    print('cfp_fp: accuracy ', accuracy)

                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_all(conf, self.agedb_30,
                                                                                   self.agedb_30_issame,
                                                                                   self.lfw, self.lfw_issame,
                                                                                   self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('agedb_30_cfp_fp_lfw', accuracy, best_threshold, roc_curve_tensor)
                    print('agedb_30_cfp_fp_lfw ', accuracy)


                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               