import os, sys
import cv2
import numpy as np

from multiprocessing import Pool
# import copy_reg
import copyreg
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    # Pii为预测正确的数量，Pij和Pji分别被解释为假正和假负，尽管两者都是假正与假负之和
    def recall(self):  # 预测为正确的像素中确认为正确像素的个数
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):  # 分割正确的像素除以总像素
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    # 雅卡尔指数，又称为交并比（IOU）
    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def get_iou(data_list, class_num, save_path=None):
    """ 
    Args:
      data_list: a list, its elements [gt, output]
      class_num: the number of label
    """
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    # print(j_list)
    # print(M)
    # print('meanIOU: ' + str(aveJ) + '\n')

    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list
