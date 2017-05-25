# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06


# This script provides a class to generate dataset
# for training and validating the model.


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


class Dataset():

    def __init__(self):
        '''__INIT__

        Initialization of the instance.

        '''

        self.T1 = []        # T1 volume
        self.T2 = []        # T2 volume
        self.PD = []        # PD volume
        self.GT = []        # Ground truth volume

        self.cn = 3         # Number of groups
        self.fn = 3         # Number of features

        self.train = []     # Training set
        self.valid = []     # Validation set

        self.test = []      # Test set
        self.dims = []      # Data dimensions
        self.test_num = 0   # Number of test case

        return

    def load_data(self, path, name, norm=False):
        '''LOAD_DATA

        Load data from .mat files, it will be normalized
        if parameter "norm" is True.

        '''

        data = nib.load(path).get_data()
        if norm:
            data -= np.min(data)
            data /= np.max(data)

        exec('self.' + name + '= data')

        # Set the data dimensions
        if name == 'GT':
            self.dims = data.shape

        return

    def get_mask(self, GT, values, label):
        '''GET_MASK

        Generate mask for different brain tissues.

        Input values:
        1 for CSF, 2 or 8 for GM, 3 for WM.


        '''

        if type(values) == int:
            values = [values]

        mask = np.zeros(GT.shape)
        for i in range(len(values)):
            mask += (GT == values[i]) * label

        return mask

    def plot_slice(self, data, idx):
        '''PLOT_SLICE

        Show one slice according to the given index.

        '''

        plt.figure()
        plt.imshow(data[idx, :, :], cmap='gray')
        plt.axis('off')
        plt.show()

        return

    def extract_data(self, GT_mask, s_idx):
        '''EXTRACT_DATA

        On the basis of given indices of slices, extract
        point value from T1, T2 and PD to for feature matrix.
        Each point has 5 dimensions of information:
        index (position in slice), T1 value, T2 value, PD value,
        label (1 for CSF, 2 for GM, 3 for WM).

        '''

        data = np.array([]).reshape((-1, self.cn + 2))
        for i in range(1, self.cn + 1):
            mask = (GT_mask == i)
            idx = np.where(mask.flatten())[0].reshape((-1, 1))
            T1 = self.T1[s_idx, :, :][mask].reshape((-1, 1))
            T2 = self.T2[s_idx, :, :][mask].reshape((-1, 1))
            PD = self.PD[s_idx, :, :][mask].reshape((-1, 1))

            label = np.ones(T1.shape) * i

            one_class = np.hstack((idx, T1, T2, PD, label))
            data = np.vstack((data, one_class))

        # Reorder the data randomly
        new_idx = np.random.permutation(data.shape[0])

        return data[new_idx, :]

    def group_data(self, GT_mask, prop=[0.6, 0.2, 0.2], reserve=90):
        '''GROUP_DATA

        Generate training and validation data according to
        the known proportion, default value is [0.6, 0.2, 0.2],
        which means that points extracted from 60% slices are
        training data, points extracted from 20% slices are
        validation data, the other points in left 20% slices
        are testing data. Reserve one slice for test.

        '''

        slice_num = GT_mask.shape[2]
        slice_idx = np.random.permutation(slice_num)

        tn = int(np.round(slice_num * prop[0]))
        vn = int(np.round(slice_num * prop[1]))

        trs_idx = slice_idx[:tn]
        vas_idx = slice_idx[tn:(tn + vn - 1)]
        tes_idx = slice_idx[(tn + vn):]

        # Reserve the 90th slice (default) for test
        if np.isnan(np.where(tes_idx == reserve)[0]):
            trs_idx[np.where(trs_idx == reserve)] = []
            vas_idx[np.where(vas_idx == reserve)] = []
            tes_idx = np.append(tes_idx, reserve)

        # Form the training set and validation set
        self.train = self.extract_data(GT_mask[trs_idx, :, :], trs_idx)
        self.valid = self.extract_data(GT_mask[vas_idx, :, :], vas_idx)

        # Save the index of test slices into a text file
        f = open('idx.txt', 'w')
        for item in tes_idx:
            f.write("{}\n".format(item))

        return

    def test_data(self, GT_mask, idx):
        '''TEST_DATA

        Generate test set from a indicated slice.

        '''

        print("You have chosen NO.{} slice.\n".format(idx))
        self.test = self.extract_data(GT_mask[idx, :, :], idx)

        return
