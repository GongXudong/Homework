# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sip import array


def get_data(mean, variance, number):
    '''
    生成二维正态分布的采样
    :param mean: 均值
    :param variance: 协方差矩阵
    :param number: 采样数
    :return: 采样点的集合
    '''
    R = cholesky(variance)
    s = np.dot(np.random.randn(number, 2), R) + mean
    return s


def show_data(s1, s2):
    plt.subplot(111)
    # 注意绘制的是散点图，而不是直方图
    plt.plot(s1[:, 0], s1[:, 1], '+', color='blue')
    plt.plot(s2[:, 0], s2[:, 1], 'o', color='purple')
    plt.show()


def show_data_with_line(data, flag, w, b):
    '''

    :param data: 二元组的集合，表示坐标
    :param flag: 与data对应的标签
    :param w: 表示一条直线
    :param b: 表示一条直线
    :return: no return
    '''
    plt.subplot(111)

    i = 0
    for item in data:
        if(flag[i][0] == 1.):
            plt.plot(item[0], item[1], '+', color='blue')
        else:
            plt.plot(item[0], item[1], '*', color='green')
        i += 1

    x = np.linspace(0, 10, 1000)
    y = (-w[0] * x - b)/w[1]
    plt.plot(x, y, color='red')
    plt.show()


def shuffle_data(arr1, arr2):
    '''
    将arr1与arr2合并，并且打乱顺序
    :param arr1:
    :param arr2:
    :return:
    '''
    tt = np.concatenate((arr1, arr2))
    np.random.shuffle(tt)
    return tt


def save_array_to_file(arr1, arr2):
    '''
    将arr1和arr2保存为文件
    :param arr1:
    :param arr2:
    :return:
    '''
    np.save('./test_x.npy', arr1)
    np.save('./test_y.npy', arr2)


def generate_data(number_train, number_validation, number_test):
    '''
    生成实验用的数据
    :param number_train:
    :param number_validation:
    :param number_test:
    :return:
    '''
    d1 = get_data([[6, 3]], [[1, 0], [0, 1]], number_train + number_validation + number_test)
    d2 = get_data([[3, 6]], [[1, 0], [0, 1]], number_train + number_validation + number_test)
    # 两个tmp数组包含标签信息
    tmp1 = np.zeros(shape=(number_train, 3))
    tmp2 = np.zeros(shape=(number_train, 3))

    i = 0
    for item in d1[0:number_train]:
        tmp1[i] = np.concatenate((item, [1]))
        i += 1
    i = 0
    for item in d2[0:number_train]:
        tmp2[i] = np.concatenate((item, [0]))
        i += 1

    tmp = shuffle_data(tmp1, tmp2)
    # 训练数据
    data_for_train = tmp[:, 0:2]
    # 训练数据对应的标签
    flag_for_train = tmp[:, 2:3]


    validation_data_A = d1[number_train: number_train + number_validation]
    test_data_A = d1[number_train + number_validation: ]

    validation_data_B = d2[number_train: number_train + number_validation]
    test_data_B = d2[number_train + number_validation:]

    # 验证用的数据
    data_for_val = np.concatenate((validation_data_A, validation_data_B))
    # 验证用的数据对应的标签
    flag_for_val = []
    for i in range(number_validation):
        flag_for_val.append([1.])
    for i in range(number_validation):
        flag_for_val.append([0.])
    flag_for_train = np.array(flag_for_train)

    # 测试用的数据
    data_for_test = np.concatenate((test_data_A, test_data_B))
    # 测试用的数据对应的标签
    flag_for_test = []
    for i in range(number_test):
        flag_for_test.append([1.])
    for i in range(number_test):
        flag_for_test.append([0.])
    flag_for_train = np.array(flag_for_train)

    save_array_to_file(data_for_test, flag_for_test)

    return data_for_train, flag_for_train, data_for_val, flag_for_val, data_for_test, flag_for_test


def model(X):

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", initializer=tf.random_normal(shape=[2, 1]))
        b = tf.get_variable("b", initializer=tf.random_normal(shape=[1]))

    m = tf.matmul(X, w) + b
    rt = tf.nn.sigmoid(m, name='predict')
    print(rt)
    return rt


def train_model(X):
    m = model(X)
    return m


def test_model(X):
    m = model(X)
    return m


def train_graph(X, Y):
    y = train_model(X)
    loss = tf.reduce_mean(- Y * tf.log(y) - (1 - Y) * tf.log(1 - y))
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=y)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    return train


def test_graph(X, Y):
    y = test_model(X)
    loss = tf.reduce_mean(- Y * tf.log(y) - (1 - Y) * tf.log(1 - y))
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=y)
    return y, loss


def work():

    data_for_train, flag_for_train, data_for_val, flag_for_val, data_for_test, flag_for_test = generate_data(100, 30, 30)

    X_train = tf.placeholder(tf.float32, shape=[None, 2], name='x_train')
    Y_train = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
    train = train_graph(X_train, Y_train)

    X_val = tf.placeholder(tf.float32, shape=[None, 2], name='x_val')
    Y_val = tf.placeholder(tf.float32, shape=[None, 1], name='y_val')
    pred, validation = test_graph(X_val, Y_val)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(500):
            # batch_size=20, 每次取20个数据点
            s = (epoch*20) % 200
            e = ((epoch + 1) * 20) % 200
            if e == 0:
                e = 200
            sess.run(train, feed_dict={X_train: data_for_train[s: e],
                                       Y_train: flag_for_train[s: e]})
            if epoch % 100 == 99:
                cur_loss, cur_pred = sess.run([validation, pred], feed_dict={X_val: data_for_val,
                                                             Y_val: flag_for_val})

                i = 0
                error = 0
                for item in flag_for_val:
                    if item - cur_pred[i] > 0.5 or cur_pred[i] - item > 0.5:
                        error += 1
                    i += 1
                acc = (len(flag_for_val) - error) * 1.0 / len(flag_for_val)

                print("epoch{%d}, loss: %s, acc: %s" % (epoch+1, cur_loss, acc))

                saver.save(sess, save_path='./save/my_model',
                           global_step=epoch+1)

        # 显示测试集与训练出的分类器
        with tf.variable_scope("model", reuse=True):
            ww = tf.get_variable('w')
            bb = tf.get_variable('b')
            [www, bbb] = sess.run([ww, bb])
            print(www, bbb)
            show_data_with_line(data_for_val, flag_for_val, www, bbb)


if __name__ == "__main__":
    work()

