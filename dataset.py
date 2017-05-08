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

        self.cn = 3
        self.fn = 3

        self.train = []
        self.valid = []

        self.test = []
        self.test_num = 0
        self.dims = []

        return

    def load_data(self, path, name, norm=False):
        '''
        '''

        data = sio.loadmat(path)[name]
        if norm:
            data -= np.min(data)
            data /= np.max(data)

        exec('self.' + name + '= data')

        if name == 'GT':
            self.dims = data.shape

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

    def extract_data(self, GT_mask, s_idx):
        '''
        '''

        data = np.array([]).reshape((-1, self.cn + 2))
        for i in range(1, self.cn + 1):
            mask = (GT_mask == i)
            idx = np.where(mask.flatten())[0].reshape((-1, 1))
            T1 = self.T1[:, :, s_idx][mask].reshape((-1, 1))
            T2 = self.T2[:, :, s_idx][mask].reshape((-1, 1))
            PD = self.PD[:, :, s_idx][mask].reshape((-1, 1))

            label = np.ones(T1.shape) * i

            one_class = np.hstack((idx, T1, T2, PD, label))
            data = np.vstack((data, one_class))

        new_idx = np.random.permutation(data.shape[0])

        return data[new_idx, :]

    def group_data(self, GT_mask, prop=[0.6, 0.2, 0.2]):
        '''
        '''

        slice_num = GT_mask.shape[2]
        slice_idx = np.random.permutation(slice_num)

        tn = int(np.round(slice_num * prop[0]))
        vn = int(np.round(slice_num * prop[1]))

        trs_idx = slice_idx[:tn]
        vas_idx = slice_idx[tn:(tn + vn - 1)]
        tes_idx = slice_idx[(tn + vn):]

        self.train = self.extract_data(GT_mask[:, :, trs_idx], trs_idx)
        self.valid = self.extract_data(GT_mask[:, :, vas_idx], vas_idx)

        f = open('idx.txt', 'w')
        for item in tes_idx:
            f.write("{}\n".format(item))

        return

    def test_data(self, GT_mask, idx):
        '''
        '''

        # f = open('idx.txt', 'r')
        # tes_idx = [int(l.split('\n')[0]) for l in f.readlines()]

        self.test = self.extract_data(GT_mask[:, :, idx], idx)
        # self.test_num = len(idx)

        return  # np.array(tes_idx)
