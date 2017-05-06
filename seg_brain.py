# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06
#


import numpy as np
from dataset import Dataset


T1_path = 'Data/T1.mat'
T2_path = 'Data/T2.mat'
PD_path = 'Data/PD.mat'
GT_path = 'Data/GT.mat'

ds = Dataset()

ds.load_data(T1_path, 'T1')
ds.load_data(T2_path, 'T2')
ds.load_data(PD_path, 'PD')
ds.load_data(GT_path, 'GT')

# ds.plot_slice(GT, 90)

CSF_mask = ds.get_mask(ds.GT, values=1, label=1)
GM_mask = ds.get_mask(ds.GT, values=[2, 8], label=2)
WM_mask = ds.get_mask(ds.GT, values=3, label=3)
GT_mask = CSF_mask + GM_mask + WM_mask

# ds.plot_slice(GT_mask, 90)

class_num = 3
ds.extract_data(GT_mask, class_num)

# print(ds.data.shape)
# print(np.unique(ds.data[:, 3]))

ds.group_data()

# print(np.where(ds.train[:, -1] == 1)[0].shape)
# print(np.where(ds.train[:, -1] == 2)[0].shape)
# print(np.where(ds.train[:, -1] == 3)[0].shape)

# print(np.where(ds.valid[:, -1] == 1)[0].shape)
# print(np.where(ds.valid[:, -1] == 2)[0].shape)
# print(np.where(ds.valid[:, -1] == 3)[0].shape)

# print(np.where(ds.test[:, -1] == 1)[0].shape)
# print(np.where(ds.test[:, -1] == 2)[0].shape)
# print(np.where(ds.test[:, -1] == 3)[0].shape)


