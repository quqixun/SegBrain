# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06
#


import numpy as np
import tensorflow as tf
import tensorlayer as tl


class Model():

    def __init__(self, dataset):
        '''
        '''

        self.dataset = dataset

        return

    def hidden_layer(self, net, n_units=100, stddev=1e-4, name='hl'):
        W = tf.truncated_normal_initializer(stddev=stddev)
        b = tf.truncated_normal_initializer(stddev=stddev)

        net = tl.layers.DenseLayer(net, n_units=n_units, act=tf.identity,
                                   W_init=W, b_init=b, name=name)
        return net

    def build_network(self, x):
        '''
        '''

        net = tl.layers.InputLayer(x, name='il')
        net = self.hidden_layer(net, n_units=100,
                                stddev=1 / (3 * 100), name='hl1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn1')
        net = self.hidden_layer(net, n_units=100,
                                stddev=1 / (100 * 100), name='hl2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn2')
        net = self.hidden_layer(net, n_units=3,
                                stddev=1 / (100 * 3), name='ol')

        return net

    def train_model(self, epochs=10, iters=100,
                    batch_size=100, learning_rate=1e-3):
        '''
        '''

        x = tf.placeholder(tf.float32, [batch_size, self.dataset.class_num])
        y = tf.placeholder(tf.float32, [batch_size, self.dataset.class_num])

        net = self.build_network(x)
        y_out = net.outputs

        return
