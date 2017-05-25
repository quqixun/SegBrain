# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/07

# This script is to test a new slice
# with the trained model.


import numpy as np
import tensorflow as tf
import tensorlayer as tl
from train_model import TrainModel
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class TestModel():

    def __init__(self, data):
        '''__INIT__

        Initialization of instance.

        '''

        self.data = data.test  # Test data
        self.dims = data.dims  # Dimensions of test data
        self.fn = data.fn      # The number of features
        self.cn = data.cn      # The number of classes

        self.pred = []         # Predicted labels
        self.true = []         # Real labels

        return

    def similar_coefficient(self, y_true, y_pred, method='di'):
        '''SIMILAR_COEFFICIENT

        Evaluate similarity between predicted labels and
        real labels. This function provides two method:
        dice index and jaccard index.

        '''

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        sc = np.zeros(self.cn)
        mt_str = ''

        # Calculate similar coefficient
        for i in range(self.cn):
            if method == 'di':
                sc[i] = 2 * cm[i, i] / (np.sum(cm[i, :] + cm[:, i]))
                mt_str = 'Dice Index'
            elif method == 'ji':
                sc[i] = cm[i, i] / ((np.sum(cm[i, :] + cm[:, i])) - cm[i, i])
                mt_str = 'Jaccard Index'
            else:
                print("Wrong method!")
                return None

        # Print the result
        print("\n" + mt_str + ":")
        print("CSF({0:.4f})  GM({1:.4f})  WM({2:.4f})\n"
              .format(sc[0], sc[1], sc[2]))

        return sc

    def test_model(self, model='model.npz'):
        '''TEST_MODEL

        Segment a new brain slice with the trained model,
        getting predicted labels and calculate similar coefficient.

        '''

        if self.data.shape[0] == 0:
            print("Test slice - This slice has no CSF, GM and WM.")
            return

        # Computation graph
        data_num = self.data.shape[0]
        x = tf.placeholder(tf.float32, shape=[data_num, self.fn])

        # Obtain the result form the model
        net = TrainModel().build_network(x)
        y_out = net.outputs

        # Obtain the predicted labels
        pred = tf.reshape(tf.argmax(y_out, 1) + 1, shape=[data_num])

        # Assign model's weights with the saved parameters
        sess = tf.Session()
        params = tl.files.load_npz(name=model)
        tl.files.assign_params(sess, params, net)

        y_pred = sess.run(pred, feed_dict={x: self.data[:, 1:-1]})
        y_pred = y_pred.reshape((-1, 1))
        y_true = self.data[:, -1].reshape((-1, 1)).astype(int)

        # Calculate similar coefficient
        self.similar_coefficient(y_true, y_pred, 'di')
        self.similar_coefficient(y_true, y_pred, 'ji')

        sess.close()

        self.pred = y_pred
        self.true = y_true

        return

    def compare_slice(self):
        '''COMPARE_SLICES

        Plot real slice, predicted slice and the wrong cases
        in the slice.

        '''

        if self.data.shape[0] == 0:
            print("Plot slice - This slice has no CSF, GM and WM.")
            return

        v_shape = self.dims[1:3]
        pred_v = np.zeros(v_shape).reshape((-1, 1))
        true_v = np.zeros(v_shape).reshape((-1, 1))

        idx = self.data[:, 0].astype(int)
        pred_v[np.array(idx)] = self.pred
        true_v[np.array(idx)] = self.true

        pred_v = pred_v.reshape(v_shape)
        true_v = true_v.reshape(v_shape)

        # Extract errors and determine their locations in slice
        errors = np.where(pred_v != true_v)
        x, y = errors[0], errors[1]

        plt.figure()
        # Plot real slice
        plt.subplot(1, 3, 1)
        plt.imshow(true_v, cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth', fontsize=22)
        # Plot predicted slice
        plt.subplot(1, 3, 2)
        plt.imshow(pred_v, cmap='gray')
        plt.axis('off')
        plt.title('Prediction', fontsize=22)
        # Plot error
        plt.subplot(1, 3, 3)
        plt.imshow(true_v, cmap='gray')
        plt.scatter(y, x, s=3, c='r')
        plt.axis('off')
        plt.title('Errors ({0}/{1})'.format(len(x), len(self.pred)),
                  fontsize=22)

        # Display the plot in full screen
        fig = plt.get_current_fig_manager()
        fig.window.showMaximized()
        plt.show()

        return
