# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06
#


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class Dataset():

    def __init__(self):
        '''
        '''

        self.T1 = []
        self.T2 = []
        self.PD = []
        self.GT = []

        self.data = []
        self.class_num = 0

        self.test = []
        self.train = []
        self.valid = []

        return

    def load_data(self, path, name):
        '''
        '''

        data = sio.loadmat(path)[name]
        exec('self.' + name + '= data')

        return

    def get_mask(self, GT, values, label):
        '''
        '''

        if type(values) == int:
            values = [values]

        mask = np.zeros(GT.shape)
        for i in range(len(values)):
            mask += (GT == values[i]) * label

        return mask

    def plot_slice(self, data, no):
        '''
        '''

        plt.figure()
        plt.imshow(data[:, :, no], cmap='gray')
        plt.axis('off')
        plt.show()
        return

    def plot_scatters(self):
        '''
        '''
        return

    def extract_data(self, GT_mask, class_num):
        '''
        '''

        self.class_num = class_num
        self.data = np.array([]).reshape((-1, class_num + 1))
        for i in range(1, class_num + 1):
            T1 = self.T1[GT_mask == i].reshape((-1, 1))
            T2 = self.T2[GT_mask == i].reshape((-1, 1))
            PD = self.PD[GT_mask == i].reshape((-1, 1))

            label = np.ones(T1.shape) * i

            one_class = np.hstack((T1, T2, PD, label))
            self.data = np.vstack((self.data, one_class))

        return

    def group_data(self, prop=[0.6, 0.2, 0.2]):
        '''
        '''

        self.test = np.array([]).reshape((-1, self.class_num + 1))
        self.train = np.array([]).reshape((-1, self.class_num + 1))
        self.valid = np.array([]).reshape((-1, self.class_num + 1))

        for i in range(1, self.class_num + 1):
            idx = np.where(self.data[:, -1] == i)[0]
            data_num = idx.shape[0]

            tn_pos = int(np.round(data_num * prop[0]))
            vt_pos = int(np.round(data_num * prop[1])) + tn_pos
            # tt_pos = int(np.round(data_num * prop[2])) + vt_pos

            idx = np.random.permutation(idx)
            self.train = np.vstack((self.train, self.data[idx[:tn_pos], :]))
            self.valid = np.vstack((self.valid, self.data[idx[tn_pos:vt_pos - 1], :]))
            self.test = np.vstack((self.test, self.data[idx[vt_pos:data_num - 1], :]))

        return
