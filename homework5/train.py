#--coding=utf-8---
import tensorflow as tf
import numpy as np
from lenet import LeNet
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
LOGDIR = '.'

def pad_data(x):
    x = np.reshape(x, (-1, 28, 28, 1))
    return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 1], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y')

    x_image = tf.reshape(x, [-1, 32, 32, 1])
    tf.summary.image('input', x_image, 3)

    lenet = LeNet(x, y_, 0., 0.01)
    train_op = lenet.get_train_op()
    loss, acc = lenet.get_loss_and_acc()

    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'summary'))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(5000):

            x_train, y_train = mnist.train.next_batch(50, shuffle=True)
            x_train = pad_data(x_train)

            if epoch % 5 == 0:
                s = sess.run(merged_summary, feed_dict={x: x_train, y_: y_train})
                writer.add_summary(s, epoch)

            sess.run(train_op, feed_dict={x: x_train, y_: y_train})

            if epoch % 1000 == 999:
                x_val, y_val = mnist.validation.next_batch(5000)
                x_val = pad_data(x_val)
                loss_, acc_ = sess.run([loss, acc], feed_dict={x: x_val, y_: y_val})
                print('epoch{%d}, loss: %lf, acc: %lf' % ((epoch + 1), loss_, acc_))
                saver.save(sess, save_path="./save/lenet_save", global_step=epoch+1)

        writer.flush()
        writer.close()

        x_test, y_test = mnist.test.next_batch(10000)
        x_test = pad_data(x_test)
        acc__ = sess.run(acc, feed_dict={x: x_test, y_: y_test})
        print("Test acc: ", acc__)
