# Created by Qixun Qu
# quqixun@gmail.com
# 2017/05/06


# This script it to buuld the model to be trained.


import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt


class TrainModel():

    def __init__(self, data=[]):
        '''__INIT__

        Initialization of the instance.

        '''

        self.data = data    # Test set

        return

    def dense_layer(self, net, n_units=100, stddev=1e-4, name='hl'):
        '''DENSE_LAYER

        Generate parameterd for two hidden layers and
        the output layer.

        '''
        # Initialize the weights and bias
        W = tf.truncated_normal_initializer(stddev=stddev)
        b = tf.truncated_normal_initializer(stddev=stddev)

        net = tl.layers.DenseLayer(net, n_units=n_units, act=tf.identity,
                                   W_init=W, b_init=b, name=name)

        return net

    def build_network(self, x):
        '''BUILD_NETWORK

        Net structure:
        input:              500 x 3 (training batch)
        1st hidden layer:   256 neurons, 3 x 256 weights
        2nd hidden layer:   256 neurons, 256 x 256 weights
        output layer:       3 neurons, 256 x 3 weights
        output:             500 x 3 (3 classes)

        '''

        net = tl.layers.InputLayer(x, name='il')
        net = self.dense_layer(net, 256, 1 / (3 * 256), 'hl1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn1')
        net = self.dense_layer(net, 256, 1 / (256 * 256), 'hl2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, name='bn2')
        net = self.dense_layer(net, 3, 1 / (256 * 3), 'ol')

        return net

    def reshape_labels(self, labels):
        '''RESHAOE_LABEL

        Original label formation, for example: [1, 2, 3].

        Redhaped label of the example:
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].

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
        '''GET_FD

        Generate feed dictionary which is to be input into
        the model to train the model or evaluate by validating.

        '''

        num = data.shape[0]
        idx = np.random.permutation(num)
        features = data[idx[:batch_size], 1:-1]
        labels = data[idx[:batch_size], -1]

        return features, self.reshape_labels(labels)

    def train_model(self, epochs=10, iters=100,
                    batch_size=100, learning_rate=1e-3):
        '''TRAIN_MODEL

        Main function to train the model.
        Input parameters are basic settings for training process.

        '''

        # Computation graph
        x = tf.placeholder(tf.float32, [batch_size, self.data.fn])
        y = tf.placeholder(tf.float32, [batch_size, self.data.cn])

        net = self.build_network(x)

        # Obtain the net output
        y_out = net.outputs
        y_out = tf.reshape(y_out, shape=[batch_size, self.data.cn])

        # Compute loss function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))

        # Define the optimizer
        op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Calculate classification accuracy
        y_arg = tf.reshape(tf.argmax(y_out, 1), shape=[batch_size])
        correct_prediction = tf.equal(y_arg, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()

        init = tf.group(tf.local_variables_initializer(),
                        tf.global_variables_initializer())

        sess.run(init)

        # Initialize vectors to save results of each training step
        all_loss = np.zeros((epochs, 2))
        all_accu = np.zeros((epochs, 2))

        # Run training iteration, in each iteration, print the result
        # and put it into the vector
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

        # Save model into a file for reusing
        tl.files.save_npz(net.all_params, 'model.npz')
        sess.close()

        # Plot learning curve
        self.plot_loss_accu(all_loss, all_accu)

        return

    def plot_loss_accu(self, loss, accu):
        '''PLOT_LOSS_ACCU

        Plot loss and accuracy of both training and validation
        process with respect to iterations.

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
