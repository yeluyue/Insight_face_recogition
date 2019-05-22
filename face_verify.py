import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true", default=False)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces', default=1.2, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true", default=True)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=True)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true", default=True)
    args = parser.parse_args()

    conf = get_config(False)
    # 设置加载模型类型
    conf.use_mobilfacenet = False
    conf.net_mode = 'facenet20'
    # 设置模型保存地址
    conf.save_path = conf.work_path + 'save'
    model_name = 'model_2019-04-16-15-57_accuracy:0.876_step:121480_final.pth'
    # 使用mtcnn检测人脸
    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold

    if conf.device.type == 'cpu':
        learner.load_state(conf, model_name, True, True)
    else:
        learner.load_state(conf, model_name, True, True)
    learner.model.eval()
    print('learner loaded')
    # 准备图像检索库，使用mtcnn进行人脸提取和对齐，预先提取人脸特征
    # 每张图想最多检测10张人脸，检测的最小人脸是30
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    if args.save:
        # 设置摄像头读取的视频参数，读取的图像大小是720*1280*3
        video_writer = cv2.VideoWriter(conf.data_path + '/recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280, 720))
        # frame rate 6 due to my laptop is quite slow...
    while cap.isOpened():
        # 打开摄像头
        isSuccess, frame = cap.read()
        if isSuccess:            
            try:
#                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:, :-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1] # personal choice
                results, score = learner.infer(conf, faces, targets, args.tta)
                for idx, bbox in enumerate(bboxes):
                    if args.score:
                        # score代表是待识别图像与人脸检索库人脸数据最小的距离
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                print('detect error')    
                
            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()    