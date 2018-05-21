#--coding=utf-8---
import tensorflow as tf
import numpy as np
from lenet import LeNet
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

def pad_data(x):
    x = np.reshape(x, (-1, 28, 28, 1))
    return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])

    lenet = LeNet(x, y_, 0., 0.01)

    train_op = lenet.get_train_op()
    loss, acc = lenet.get_loss_and_acc()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(5000):
            x_train, y_train = mnist.train.next_batch(50, shuffle=True)
            x_train = pad_data(x_train)
            sess.run(train_op, feed_dict={x: x_train, y_: y_train})

            if epoch % 500 == 499:
                x_val, y_val = mnist.validation.next_batch(5000)
                x_val = pad_data(x_val)
                loss_, acc_ = sess.run([loss, acc], feed_dict={x: x_val, y_: y_val})
                print('epoch{%d}, loss: %lf, acc: %lf' % ((epoch + 1), loss_, acc_))

        x_test, y_test = mnist.test.next_batch(10000)
        x_test = pad_data(x_test)
        acc__ = sess.run(acc, feed_dict={x: x_test, y_: y_test})
        print("Test acc: ", acc__)
