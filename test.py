# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/07
#


import numpy as np
from dataset import Dataset
from test_model import TestModel


# Load data for testing model
T1_path = 'Data/T1.mat'
T2_path = 'Data/T2.mat'
PD_path = 'Data/PD.mat'
GT_path = 'Data/GT.mat'

ds = Dataset()

ds.load_data(T1_path, 'T1', norm=True)
ds.load_data(T2_path, 'T2', norm=True)
ds.load_data(PD_path, 'PD', norm=True)
ds.load_data(GT_path, 'GT')

# Generate training and validation data
CSF_mask = ds.get_mask(ds.GT, values=1, label=1)
GM_mask = ds.get_mask(ds.GT, values=[2, 8], label=2)
WM_mask = ds.get_mask(ds.GT, values=3, label=3)
GT_mask = CSF_mask + GM_mask + WM_mask

# ds.plot_slice(GT_mask, 90)

idx = ds.test_data(GT_mask, 'idx.txt')
idx_median = int(np.median(idx))
idx_pos = np.where(idx == idx_median)[0]

if len(idx_pos) == 0:
    slice_no = np.where(idx == (idx_median + 1))[0]
else:
    slice_no = idx_pos

# Test model
tm = TestModel(ds)
tm.test_model()
tm.compare_slice(slice_no)
