# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06
#


import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt


class TrainModel():

    def __init__(self, data=[]):
        '''
        '''

        self.data = data

        return

    def dense_layer(self, net, n_units=100, stddev=1e-4, name='hl'):
        W = tf.truncated_normal_initializer(stddev=stddev)
        b = tf.truncated_normal_initializer(stddev=stddev)

        net = tl.layers.DenseLayer(net, n_units=n_units, act=tf.identity,
                                   W_init=W, b_init=b, name=name)
        return net

    def build_network(self, x):
        '''
        '''

        net = tl.layers.InputLayer(x, name='il')
        net = self.dense_layer(net, 256, 1 / (3 * 256), 'hl1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn1')
        net = self.dense_layer(net, 256, 1 / (256 * 256), 'hl2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn2')
        net = self.dense_layer(net, 3, 1 / (256 * 3), 'ol')

        return net

    def reshape_labels(self, labels):
        '''
        '''

        def sub2ind(shape, rows, cols):
            return (rows * shape[1] + cols - 1).astype(int)

        bs = labels.shape[0]
        ls = [bs, len(np.unique(labels))]
        lm = np.zeros(ls).flatten()
        idx = sub2ind(ls, np.arange(bs), np.reshape(labels, [1, bs]))

        lm[idx] = 1

        return np.reshape(lm, ls)

    def get_fd(self, data, batch_size):
        '''
        '''

        num = data.shape[0]
        idx = np.random.permutation(num)
        features = data[idx[:batch_size], 1:-1]
        labels = data[idx[:batch_size], -1]

        return features, self.reshape_labels(labels)

    def train_model(self, epochs=10, iters=100,
                    batch_size=100, learning_rate=1e-3):
        '''
        '''

        x = tf.placeholder(tf.float32, [batch_size, self.data.fn])
        y = tf.placeholder(tf.float32, [batch_size, self.data.cn])

        net = self.build_network(x)

        y_out = net.outputs
        y_out = tf.reshape(y_out, shape=[batch_size, self.data.cn])

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))

        op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        y_arg = tf.reshape(tf.argmax(y_out, 1), shape=[batch_size])
        correct_prediction = tf.equal(y_arg, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()

        init = tf.group(tf.local_variables_initializer(),
                        tf.global_variables_initializer())

        sess.run(init)

        all_loss = np.zeros((epochs, 2))
        all_accu = np.zeros((epochs, 2))

        for epoch in range(epochs):
            for i in range(iters):
                t_set, t_lbl = self.get_fd(self.data.train, batch_size)
                fd_train = {x: t_set, y: t_lbl}

                sess.run(op, feed_dict=fd_train)
                time.sleep(0.05)

            v_set, v_lbl = self.get_fd(self.data.valid, batch_size)
            fd_valid = {x: v_set, y: v_lbl}

            t_loss = loss.eval(feed_dict=fd_train)
            t_accu = accuracy.eval(feed_dict=fd_train)

            v_loss = loss.eval(feed_dict=fd_valid)
            v_accu = accuracy.eval(feed_dict=fd_valid)

            print("----------\nEpoch {}\n----------".format(epoch + 1))
            print("Training loss: {0:.6f}     Trainging accu: {1:.6f}"
                  .format(t_loss, t_accu))
            print("Validation loss: {0:.6f}   Validation accu: {1:.6f}\n"
                  .format(v_loss, v_accu))

            all_loss[epoch, :] = np.array([t_loss, v_loss])
            all_accu[epoch, :] = np.array([t_accu, v_accu])

        tl.files.save_npz(net.all_params, 'model.npz')
        sess.close()

        self.plot_loss_accu(all_loss, all_accu)

        return

    def plot_loss_accu(self, loss, accu):
        '''
        '''

        x = np.arange(loss.shape[0]) + 1

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(x, loss[:, 0], label='Training')
        plt.plot(x, loss[:, 1], label='Validation')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, accu[:, 0], label='Training')
        plt.plot(x, accu[:, 1], label='Validation')
        plt.title('Accuracy')
        plt.legend()

        plt.show()

        return
