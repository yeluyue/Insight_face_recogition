'''
Use to draw ROC curve and socre transformation
'''

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
class RocCurve():

    def make_roc_curve(self, result_path):

        if result_path.endswith(".txt"):
            f = open(result_path, 'r')
            data = f.read().splitlines()
            f.close()
            result = []
            for line in data:
                distance = float(line.split(' ')[0])
                label = float(line.split(' ')[1])
                result.append([distance, label])
            result = np.array(result)
            output_figure_path = result_path.replace('.txt', '_ROC.png')
            title = os.path.basename(output_figure_path).split('.png')[0]
            roc_info, info_dict = self.draw_roc(-result[:, 0], result[:, 1], output_figure_path, title)
            return roc_info, info_dict

        elif result_path.endswith(".npy"):
            result = np.load(result_path)
            output_figure_path = result_path.replace('.npy', '_ROC.png')
            title = os.path.basename(output_figure_path).split('.png')[0]
            roc_info, info_dict = self.draw_roc(-result[:, 0], result[:, 1], output_figure_path, title)
            return roc_info, info_dict

    def draw_roc(self, score,label, output_figure_path, title=None):

        fpr, tpr, thresholds =  roc_curve(label,score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        roc_info = []
        info_dict = {}

        # 10**-6,10**-5, 10**-4, 10**-3, 10**-2, 10**-1
        for tolerance in [10 ** -7, 10 ** -6, 5e-6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]:

            #find index according to tolerance
            fpr = np.around(fpr, decimals=7)
            index = np.argmin(abs(fpr - tolerance))
            index_all = np.where(fpr == fpr[index])
            max_acc = np.max(tpr[index_all])
            threshold = np.max(abs(thresholds[index_all]))

            #draw value on the graph
            x, y = fpr[index], max_acc
            plt.plot(x, y, 'x')
            plt.text(x, y, "({:.5f}, {:.5f}) threshold={:.3f}".format(x, y, threshold))

            #save roc info log

            temp_info = 'fpr\t{}\tacc\t{:.5f}\tthreshold\t{:.6f}'.format(tolerance, max_acc, threshold)
            roc_info.append(temp_info)
            info_dict[tolerance] = max_acc

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - {}'.format(title))

        fig.savefig(output_figure_path)
        plt.close(fig)
        return roc_info, info_dict

    def calculate_transform_score(self, info_dict):
        '''

        :param info_dict:  result from draw_roc
        :return:  transfrom score result
        '''

        score_level1 = info_dict[1e-4]
        score_level2 = info_dict[1e-2]
        score_level3 = (score_level2 + score_level1) * 1.3
        score_info = []
        keys = list(info_dict.keys())
        keys.sort()
        for tolerance in keys:
            distance = info_dict[tolerance]
            t_score = self.score_transform(score_level1, score_level2, score_level3, distance)
            tmp_score_info = "threshold\t{:.4f}\tscore\t{:.2f}\tfpr\t{}".format(distance, t_score,
                                                                        tolerance)
            score_info.append(tmp_score_info)
        return score_info

    def score_transform(self, score_level1, score_level2, score_level3, distance):
        if distance < 0.3333 * score_level1:
            return 100
        if distance >= 0.3333 * score_level1 and distance < score_level1:
            score = (1.5 * (score_level1 - distance) / score_level1 * 0.2 + 0.8) * 100
            return score
        if distance >= score_level1 and distance <= score_level2:
            score = (0.8 - (distance - score_level1) / (score_level2 - score_level1) * 0.2) * 100
            return score
        if distance > score_level2 and distance < score_level3:
            score = ((score_level3 - distance) / (score_level3 - score_level2) * 0.6) * 100
            return score
        if distance > score_level3:
            return 0


    # def draw_score_plot(self, score_level1, score_level2, score_level3, x, ):
    #
    #     y = np.array(
    #         [self.score_transform(score_level1, score_level2, score_level3, x[index]) for index in range(x.shape[0])])
    #
    #     s1x_ = round(score_level1 / 3.0, 2)
    #     s1y_ = round(self.score_transform(score_level1, score_level2, score_level3, s1x_))
    #     s1x = round(score_level1, 2)
    #     s1y = round(self.score_transform(score_level1, score_level2, score_level3, s1x))
    #     s2x = round(score_level2, 2)
    #     s2y = round(self.score_transform(score_level1, score_level2, score_level3, s2x))
    #     s3x = round(score_level3, 2)
    #     s3y = round(self.score_transform(score_level1, score_level2, score_level3, s3x))
    #
    #     info_xy = [(s1x_,s1y_), (s1x, s1y), (s2x, s2y), (s3x, s3y)]
    #     return y, info_xy
    #
    # def plot_score_transform(self):
    #     score_level1 = [2.67342,2.4358]
    #     score_level2 = [3.18632,2.67342]
    #
    #     for index in range(len(score_level1)):
    #         s1 = score_level1[index]
    #         s2 = score_level2[index]
    #         s3 = (s1 + s2)*1.2
    #         x = np.arange(0,10,0.01)
    #         y, info_xy = self.draw_score_plot(s1, s2, s3, x)
    #
            # plt.plot(x,y)
    #         for (sx, sy) in info_xy:
    #             plt.text(sx,sy, '({},{})'.format(sx, sy))
    #
    #         plt.xlim([-0.05, 10.05])
    #         plt.ylim([-0.05, 105])
    #     plt.show()


if __name__ == "__main__":
    roc = RocCurve()
    roc.plot_score_transform()









