#--coding=utf-8---

import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./dataset/mnist', one_hot=False)


def childmodel(x):
    """
    提取mnist数据特征的子模型.

    x: mnist image, shpae is [None, 784]
    """
    with tf.variable_scope('mnist', reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02), shape=(784, 500), name='W1')
        b1 = tf.get_variable(initializer=tf.constant_initializer(value=0.), shape=[500], name='b1')
        W2 = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02), shape=(500, 10), name='W2')
        b2 = tf.get_variable(initializer=tf.constant_initializer(value=0.), shape=[10], name='b2')

        l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
        return l2


def model(input1, input2):
    """
    连体mnist网络的整体模型, 复用了childmodel中的参数.

    input1: [None, 784]
    input2: [None, 784]
    output: 0-1, 两个输入不同的概率
    """
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = childmodel(input1)
        g2 = childmodel(input2)
        sub1 = tf.subtract(g1, g2, name='sub_g')
        mul1 = tf.multiply(sub1, sub1, name='mul_sub')
        ew = tf.reduce_sum(mul1, axis=1, name='sum_mul')
        ew = tf.reshape(ew, (-1, 1))
        # 此处增加了一个可学习参数theta，因为子网络的两个输出特征的L2范数总为正数，
        # 为了应用sigmoid函数，需要对其做一定的偏移，偏移量为可学习参数theta
        theta = tf.get_variable(initializer=tf.truncated_normal_initializer(0., 1.), shape=[1], name='thresh')
        pred = tf.nn.sigmoid(tf.subtract(ew, theta), name='pred')
        return pred


def prepare(input, label):
    """
    对输入数据的预处理，采取了以下措施平衡输入中正负样本的数量：
    step 1: 对于一个batch中的数据input1_1， 拷贝出一个副本input1_2，打乱input2_1的顺序，那么input1_1和input1_2中对应位置的数据label不同的概率大约为0.9;
    step 2: 对input1_1和input2_1根据label进行排序得到input1_2, input2_2，那么对应位置的数据label均一样。
    step 3: 将input1_1和input1_2拼接，input2_1和input2_2拼接得到一个新的batch.
    新的batch的size是原来的2倍，其中的正负样本的比率约为11：9.
    """
    label = np.reshape(label, (-1, 1))

    input1_1 = np.column_stack((input, label))
    input1_1 = input1_1.astype(np.float32)
    input2_1 = deepcopy(input1_1)
    np.random.shuffle(input2_1)

    input1_2 = input1_1[input1_1[:, -1].argsort()]
    input2_2 = input2_1[input2_1[:, -1].argsort()]

    input1 = np.row_stack((input1_1, input1_2))
    input2 = np.row_stack((input2_1, input2_2))
    return input1, input2


def ACC(predict, input1, input2):
    """
    计算连体网络在输入数据上的准确率.
    """
    predict = predict > 0.5
    label = 1. - (input1[:, -1:] == input2[:, -1:])
    acc = predict == label
    acc = np.mean(acc)
    return acc


x1 = tf.placeholder(tf.float32, shape=[None, 784], name='x1')
x2 = tf.placeholder(tf.float32, shape=[None, 784], name='x2')
y1 = tf.placeholder(tf.float32, shape=[None, 1], name='y1')
y2 = tf.placeholder(tf.float32, shape=[None, 1], name='y2')
y = tf.subtract(1., tf.cast(tf.equal(y1, y2), dtype=tf.float32), name='y')

pred = model(x1, x2)
# 此处对loss的计算公式做了修改，将连体网络的输出pred，即两个输入不同的概率替换原loss函数
# 中的Ew，并简化了公式中的常数项，因此新的loss函数为：
# loss = (1-y)*pred^2 + y*exp(-pred)
# 实验表明，此loss函数依然能取得90%的准确率
loss = tf.reduce_mean(
    tf.add(
        tf.multiply(tf.subtract(1., y), tf.multiply(pred, pred)),
        tf.multiply(y, tf.exp(-pred))
    ), name='loss')
# 实验表明，学习率为0.01时还是SGD效果更好
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 迭代次数：20000, batch_size: 64
    for itera in range(20000):
        input, label = mnist.train.next_batch(64)
        input1, input2 = prepare(input, label)
        l, _ = sess.run([loss, optimizer], feed_dict={x1: input1[:, :-1], x2: input2[:, :-1], y1: input1[:, -1:], y2: input2[:, -1:]})

        # 每隔2000个迭代，计算在验证集上的acc
        if (itera + 1) % 2000 == 0:
            input, label = mnist.validation.images, mnist.validation.labels
            input1, input2 = prepare(input, label)

            predict = sess.run(pred, feed_dict={x1: input1[:, :-1], x2: input2[:, :-1]})
            acc = ACC(predict, input1, input2)
            print('iteration: %d, loss: %.4f, val_acc: %.4f' % ( itera +1, l, acc))

    # 计算测试集合上的acc
    print()
    input, label = mnist.test.images, mnist.test.labels
    input1, input2 = prepare(input, label)
    predict = sess.run(pred, feed_dict={x1: input1[:, :-1], x2: input2[:, :-1]})
    acc = ACC(predict, input1, input2)
    print('test_acc: %.4f' % (acc))