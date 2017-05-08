# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/07
#


import numpy as np
import tensorflow as tf
import tensorlayer as tl
from train_model import TrainModel
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class TestModel():

    def __init__(self, data):
        '''
        '''

        self.data = data.test
        # self.test_num = data.test_num
        self.dims = data.dims
        self.fn = data.fn
        self.cn = data.cn

        self.pred = []
        self.true = []

        return

    def similar_coefficient(self, y_true, y_pred, method='di'):
        '''
        '''

        cm = confusion_matrix(y_true, y_pred)

        sc = np.zeros(self.cn)
        mt_str = ''

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

        print("\n" + mt_str + ":")
        print("CSF({0:.4f})  GM({1:.4f})  WM({2:.4f})\n"
              .format(sc[0], sc[1], sc[2]))

        return sc

    def test_model(self, model='model.npz'):
        '''
        '''

        data_num = self.data.shape[0]
        x = tf.placeholder(tf.float32, shape=[data_num, self.fn])

        net = TrainModel().build_network(x)
        y_out = net.outputs

        pred = tf.reshape(tf.argmax(y_out, 1) + 1, shape=[data_num])

        sess = tf.Session()
        params = tl.files.load_npz(name=model)
        tl.files.assign_params(sess, params, net)

        y_pred = sess.run(pred, feed_dict={x: self.data[:, 1:-1]})
        y_pred = y_pred.reshape((-1, 1))
        y_true = self.data[:, -1].reshape((-1, 1)).astype(int)

        self.similar_coefficient(y_true, y_pred, 'di')
        self.similar_coefficient(y_true, y_pred, 'ji')

        sess.close()

        self.pred = y_pred
        self.true = y_true

        return

    def compare_slice(self):
        '''
        '''

        v_shape = self.dims[0:2]
        pred_v = np.zeros(v_shape).reshape((-1, 1))
        true_v = np.zeros(v_shape).reshape((-1, 1))

        idx = self.data[:, 0].astype(int)
        pred_v[np.array(idx)] = self.pred
        true_v[np.array(idx)] = self.true

        pred_v = pred_v.reshape(v_shape)
        true_v = true_v.reshape(v_shape)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(true_v, cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth')
        plt.subplot(1, 2, 2)
        plt.imshow(pred_v, cmap='gray')
        plt.axis('off')
        plt.title('Prediction')
        plt.show()

        return
