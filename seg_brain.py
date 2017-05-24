# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06


# This script is to train the model to
# carry out the segmentation of brain.


from dataset import Dataset
from train_model import TrainModel


# Load data for training
T1_path = 'Data/T1.mat'
T2_path = 'Data/T2.mat'
PD_path = 'Data/PD.mat'
GT_path = 'Data/GT.mat'

# Initialize instance for input data
ds = Dataset()

ds.load_data(T1_path, 'T1', norm=True)
ds.load_data(T2_path, 'T2', norm=True)
ds.load_data(PD_path, 'PD', norm=True)
ds.load_data(GT_path, 'GT')

# Generate training and validation dataset
CSF_mask = ds.get_mask(ds.GT, values=1, label=1)
GM_mask = ds.get_mask(ds.GT, values=[2, 8], label=2)
WM_mask = ds.get_mask(ds.GT, values=3, label=3)
GT_mask = CSF_mask + GM_mask + WM_mask

ds.group_data(GT_mask)

# Training the model
tm = TrainModel(ds)
tm.train_model(epochs=15, iters=50,
               batch_size=500, learning_rate=3e-4)
