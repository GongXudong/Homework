#--coding=utf-8---

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
mnist = input_data.read_data_sets('./data/mnist', one_hot=False)

tf.set_random_seed(1)

print(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)


def process_data(x1, y1, x2, y2, theta=0.05):
    '''
    简单的处理数据，使正负样本大致均衡
    只能利用10%左右的数据
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param theta:
    :return:
    '''
    res_x1 = []
    res_x2 = []
    res_y1 = []
    res_y2 = []
    res_y = []
    cnt_eq = 0
    cnt_neq = 0
    theta_ = int(len(x1) * theta)
    for i in range(min(len(x1), len(x2))):
        if y1[i] == y2[i] and (abs((cnt_eq + 1) - cnt_neq) <= theta_):
            res_x1.append(x1[i])
            res_x2.append(x2[i])
            res_y1.append(y1[i])
            res_y2.append(y2[i])
            res_y.append(0)
            cnt_eq += 1
        if y1[i] != y2[i] and (abs((cnt_neq + 1) - cnt_eq) <= theta_):
            res_x1.append(x1[i])
            res_x2.append(x2[i])
            res_y1.append(y1[i])
            res_y2.append(y2[i])
            res_y.append(1)
            cnt_neq += 1

    return np.array(res_x1), np.array(res_x2), np.array(res_y)



def single_model(X):
    '''
    返回单个网络的输出
    :param X:
    :return:
    '''
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        w_1 = tf.get_variable('w1', shape=[784, 500], initializer=tf.truncated_normal_initializer(mean=0., stddev=0.05))
        b_1 = tf.get_variable('b1', shape=[500], initializer=tf.zeros_initializer(dtype=tf.float32))

        w_2 = tf.get_variable('w2', shape=[500, 10], initializer=tf.truncated_normal_initializer(mean=0., stddev=0.05))
        b_2 = tf.get_variable('b2', shape=[10], initializer=tf.zeros_initializer(dtype=tf.float32))

    m_1 = tf.add(tf.matmul(X, w_1), b_1)
    r_1 = tf.nn.relu(m_1)

    m_2 = tf.add(tf.matmul(r_1, w_2), b_2)
    r_2 = tf.nn.relu(m_2)
    return r_2


def dual_train_model(X1, X2):
    '''
    返回连体网络的预测值
    :param X1:
    :param X2:
    :return:
    '''
    m1 = single_model(X1)
    m2 = single_model(X2)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(m1, m2)), 1) + 1e-6)
    return E_w


def dual_test_model(X1, X2):
    m1 = single_model(X1)
    m2 = single_model(X2)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(m1, m2)), 1) + 1e-6)
    return E_w


def train_graph(X1, X2, Y):
    pred = dual_train_model(X1, X2)
    with tf.name_scope('train'):
        loss_1 = tf.multiply(tf.multiply(tf.subtract(tf.constant(1., dtype=tf.float32), Y),
                                         tf.constant(2. / 5., dtype=tf.float32)),
                             tf.square(pred))
        loss_2 = tf.multiply(tf.multiply(Y, tf.constant(2. * 5., dtype=tf.float32)),
                             tf.exp(tf.multiply(tf.constant(-2.77 / 5., dtype=tf.float32), pred)))
        loss = tf.reduce_mean(tf.add(loss_1, loss_2))

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train = optimizer.minimize(loss)
    return pred, train


def test_graph(X1, X2, Y):
    pred = dual_test_model(X1, X2)
    with tf.name_scope('test'):

        loss_1 = tf.multiply(tf.multiply(tf.subtract(tf.constant(1., dtype=tf.float32), Y),
                                         tf.constant(2. / 5., dtype=tf.float32)),
                             tf.square(pred))
        loss_2 = tf.multiply(tf.multiply(Y, tf.constant(2. * 5., dtype=tf.float32)),
                             tf.exp(tf.multiply(tf.constant(-2.77 / 5., dtype=tf.float32), pred)))
        loss = tf.reduce_mean(tf.add(loss_1, loss_2))

    return pred, loss


def get_acc(y, y_pred):
    y_pred = y_pred > 2.3
    acc = (y == y_pred)
    return np.mean(acc)

def work():
    # train part
    x1_train = tf.placeholder(tf.float32, shape=[None, 784], name='x1_train')
    x2_train = tf.placeholder(tf.float32, shape=[None, 784], name='x2_train')
    y_train = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
    pred_in_train, train = train_graph(x1_train, x2_train, y_train)

    # validation part
    x1_val = tf.placeholder(tf.float32, shape=[None, 784], name='x1_val')
    x2_val = tf.placeholder(tf.float32, shape=[None, 784], name='x2_val')
    y_val = tf.placeholder(tf.float32, shape=[None, 1], name='y_val')
    pred, loss = test_graph(x1_val, x2_val, y_val)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(50000):
            # prepare data for training
            x1, y1 = mnist.train.next_batch(batch_size=batch_size)
            x2, y2 = mnist.train.next_batch(batch_size=batch_size)
            x1_for_train, x2_for_train, y_tmp = process_data(x1, y1, x2, y2, 0.05)
            y_for_train = y_tmp.reshape((-1, 1))

            __, _ = sess.run([pred_in_train, train], feed_dict={x1_train: x1_for_train,
                                       x2_train: x2_for_train,
                                       y_train: y_for_train})
            # print('epoch{%d}, acc: %lf' % ((epoch+1), get_acc(y_for_train, __)))

            if epoch % 100 == 99:
                # prepare data for validation
                x1_in_val, y1_in_val = mnist.validation.next_batch(batch_size=2500)
                x2_in_val, y2_in_val = mnist.validation.next_batch(batch_size=2500)
                x1_for_val, x2_for_val, y_tmp2 = process_data(x1_in_val, y1_in_val, x2_in_val, y2_in_val, 0.05)
                y_for_val = y_tmp2.reshape((-1, 1))

                _p, cur_loss = sess.run([pred, loss], feed_dict={x1_val: x1_for_val,
                                                                 x2_val: x2_for_val,
                                                                 y_val: y_for_val})

                print('epoch{%d}, acc: %lf, loss: %lf' % ((epoch + 1), get_acc(y_for_val, _p), cur_loss))
                print("ave_dis for pos: %lf; ave_dis for neg: %lf" % (np.average(_p[y_tmp2 == 0.]), np.average(_p[y_tmp2 == 1.])))


if __name__ == '__main__':
    work()
    # print(get_acc(np.array([1,0,1,1]), np.array([0.8, 0.6, 0.6, 0.7])))
    # x1_in_val1, y1_in_val1 = mnist.train.next_batch(batch_size=27500)
    # x2_in_val1, y2_in_val1 = mnist.train.next_batch(batch_size=27500)
    # x1_in_val, y1_in_val = mnist.train.next_batch(batch_size=27500)
    # x2_in_val, y2_in_val = mnist.train.next_batch(batch_size=27500)
    # a, b, c = process_data(x1_in_val, y1_in_val, x2_in_val, y2_in_val)
    # print(len(a), len(b), len(c))

    # a = [1,2,3,4,5]
    # c = [5,4,3,2,1]
    # b = [0,1,0,0,0]
    # d = [0,1,1,1,1]
    # e,f,g = process_data(a,b,c,d,1)
    # print(e)
    # print(f)
    # print(g)
