from config import get_config
from Learner_zhu import face_learner
import argparse
import torch

torch.cuda.set_device(0)
# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    #设置训练过程参数
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=50, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet, resnet18,facenet20, shufflenet_v2]", default='shufflenet_v2', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-1, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat, lfw]",
                        default='msceleb', type=str)
    args = parser.parse_args()

    #进行模型参数和数据集地址配置
    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        if args.net_mode == 'ir' or args.net_mode == 'ir_se':
            conf.net_mode = args.net_mode
            conf.net_depth = args.net_depth
        elif args.net_mode == 'resnet18':
            conf.net_mode = args.net_mode
        elif args.net_mode == 'facenet20':
            conf.net_mode = args.net_mode
        elif args.net_mode == 'shufflenet_v2':
            conf.net_mode = args.net_mode
        elif args.net_mode == 'mobilenet_v2':
            conf.net_mode = args.net_mode
        elif args.net_mode == 'squeezenet_v2':
            conf.net_mode = args.net_mode
        else:
            print('error occur in the model setting')

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    learner = face_learner(conf)

    learner.train(conf, args.epochs)