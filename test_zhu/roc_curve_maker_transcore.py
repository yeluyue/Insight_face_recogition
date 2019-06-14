#use to draw Roc curve
#Author Zkx@__@
#Date 2018-01-10
#Update 2018-01-17 fixed some bug about accuracy
#create roc info list
#Update 2018-01-22 add transform part

import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def readScore(path2ScoreFile):
    with open(path2ScoreFile,'r') as f:
        data = f.read().splitlines()
    data = np.array([[float(datum.split()[0]),float(datum.split()[1])] for datum in data])
    return -data[:,0], data[:,1]#return distance label

def draw_roc(fpr,tpr,thresholds, output_figure_path, title=None):
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    #save the fpr acc threshold
    roc_info = []
    info_dict = {} #[tolerance] :  threshold
    for tolerance in [10**-7,10**-6,5.0*10**-6,10**-5, 10**-4,1.2*10**-4, 10**-3, 10**-2, 10**-1]:

        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        #same threshold match the multi-accuracy
        index_all = np.where(fpr == fpr[index])
        #select max accuracy
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))
        # print index
        x, y = fpr[index], max_acc
       
        plt.plot(x, y, 'x')
        plt.text(x, y, "({:.7f}, {:.7f}) threshold={:.7f}".format(x, y, threshold))
        temp_info = 'fpr\t{}\tacc\t{}\tthreshold\t{}'.format(tolerance, round(max_acc,5), threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = threshold

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {}'.format(title))
    #plt.legend(loc="lower right")
    fig.savefig(output_figure_path)
    plt.close(fig)
    return roc_info, info_dict
	
def roc_maker(score_file, save_roc_info=False):
    filename = os.path.basename(score_file).strip('.txt')
    output_figure_path = score_file.replace('.txt', '_ROC.png')
    distance, label = readScore(score_file)
    fpr, tpr, thresholds = roc_curve(label, distance, pos_label=1)

    #for debug  save the value of fpr tpr threshould
    # fpr_tpr = np.column_stack((fpr, tpr))
    # fpr_tpr_thre = np.column_stack((fpr_tpr, thresholds))
    # np.savetxt("/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-01-15/Result__smallmodel//score.txt",
    #            fpr_tpr_thre, fmt='%.15f', delimiter=",")
    ###
    roc_info, info_dict = draw_roc(fpr, tpr, thresholds, output_figure_path, filename)
	
    #save transform result
    if save_roc_info:
        roc_info_path = score_file.replace('.txt', '._roc_info.log')
        f = open(roc_info_path, 'w')
        score_level1 = info_dict[1e-4]
        score_level2 = info_dict[1e-2]
        score_level3 = (score_level2 + score_level1)*1.1

        f.write('score_level1:{} (1e-4)\nscore_level2:{} (1e-2)\nscore_level3:{} (> 1e-4+1e-2)\n\n'.format(score_level1, score_level2, score_level3))
        f.write("\nRoc info..\n")
        for line in roc_info:
            f.write("{}\n".format(line))
        f.close()
    

def folder_roc_maker(folder_path):
    for root_path, foldername, filenames in os.walk(folder_path):
        for filename in filenames:
             if filename.endswith('.txt'):
                print (filename)
        score_file = '{}/{}'.format(root_path, filename)
        roc_maker(score_file)
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please input distance file..")
    elif len(sys.argv) == 2:
    # score_file = "/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/result_v2_11_06_Face_ResNet20_fc_0.4_112x96_margin_2_6_C4_loss_iter_45000.txt"
        score_file = sys.argv[1]
        roc_maker(score_file)
    elif len(sys.argv) == 3:
        score_file = sys.argv[1]
        score_info = sys.argv[2]
        if score_info == "True": 
            score_value = True
            roc_maker(score_file, score_value)
        elif score_info == "False": 
            score_value = False
            roc_maker(score_file, score_value)
        else: 
            print("Input wrong type True or False")
        
  

