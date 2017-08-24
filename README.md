## Basic Information of Code

This is an implementation of a simple neural network to segment
brain tissues (white matter, gray matter and cerebrospinal fluid).

Data of brain volume is downloaded from [BrainWeb](http://brainweb.bic.mni.mcgill.ca/brainweb/).

**Author: Qixun Qu**

Date: 2017/05/02

## Dependencies

* python		3.5.2
* numpy		1.11.3
* nibabel		2.1.0
* tensorflow	1.0.0
* tensorlayer	1.4.1
* matplotlib	2.0.0

All functions are tested in Ubuntu 16.04.

## Run the Segmentation

Three steps to run the program.

**1.** Download brain volume from BrainWeb, including ground truth,
T1, T2 and PD data in .mnc files. Put these files into folder "Data".

**2.** In terminal, run "python seg_brain.py" to train the model. The model will be save in file model.npz.

**3.** In terminal, run "python test.py" to test a slice, segmentation results will be displayed, Dice index and Jaccard index is computed as well.

## Results

**Loss and dice index.**

<img src="https://github.com/quqixun/SegBrain/blob/master/Result/learning_curve.png" width="400">

**Segmentation result of one slice.**

<img src="https://github.com/quqixun/SegBrain/blob/master/Result/test_one_slice.png" width="600">
