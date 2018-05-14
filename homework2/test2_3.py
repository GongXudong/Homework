# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np


test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')

saver = tf.train.import_meta_graph('./save/my_model-500.meta')

with tf.Session() as sess:
    saver.restore(sess, './save/my_model-500')

    X = sess.graph.get_tensor_by_name('x_val:0')
    Y = sess.graph.get_tensor_by_name('y_val:0')

    predict = sess.graph.get_tensor_by_name('predict_1:0')

    res = sess.run([predict], feed_dict={X: test_x, Y: test_y})

    i = 0
    for item in test_x:
        print(i, item, test_y[i], res[0][i])
        i += 1

    i = 0
    error = 0
    for item in test_y:
        if res[0][i] - item > 0.5 or item - res[0][i] > 0.5:
            error += 1
        i += 1

    print('acc: %lf' % (1. - error * 1. / len(test_y)))

