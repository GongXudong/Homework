#--coding=utf-8---

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist', one_hot=False)

tf.set_random_seed(1)

print(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)


def balanced_batch(batch_x, batch_y, num_cls=10):
    batch_size = len(batch_y)
    pos_per_cls_e = round(batch_size / 2 / num_cls)

    index = batch_y.argsort()
    ys_1 = batch_y[index]
    # print(ys_1)

    num_class = []
    pos_samples = []
    neg_samples = set()
    cur_ind = 0
    for item in set(ys_1):
        num_class.append((ys_1 == item).sum())
        num_pos = pos_per_cls_e
        while (num_pos > num_class[-1]):
            num_pos -= 2
        pos_samples.extend(np.random.choice(index[cur_ind:cur_ind + num_class[-1]], num_pos, replace=False).tolist())
        neg_samples = neg_samples | (set(index[cur_ind:cur_ind + num_class[-1]]) - set(list(pos_samples)))
        cur_ind += num_class[-1]

    neg_samples = list(neg_samples)

    x1_index = pos_samples[::2]
    x2_index = pos_samples[1:len(pos_samples) + 1:2]

    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples) + 1:2])

    p_index = np.random.permutation(len(x1_index))
    x1_index = np.array(x1_index)[p_index]
    x2_index = np.array(x2_index)[p_index]

    r_x1_batch = batch_x[x1_index]
    r_x2_batch = batch_x[x2_index]
    r_y_batch = np.array(batch_y[x1_index] != batch_y[x2_index], dtype=np.float32)
    return r_x1_batch, r_x2_batch, r_y_batch


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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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
            x1, y1 = mnist.train.next_batch(batch_size=64*2)
            x1_for_train, x2_for_train, y_tmp = balanced_batch(x1, y1)
            y_for_train = y_tmp.reshape((-1, 1))

            __, _ = sess.run([pred_in_train, train], feed_dict={x1_train: x1_for_train,
                                       x2_train: x2_for_train,
                                       y_train: y_for_train})
            # print('epoch{%d}, acc: %lf' % ((epoch+1), get_acc(y_for_train, __)))

            if epoch % 1000 == 999:
                # prepare data for validation
                x1_in_val, y1_in_val = mnist.validation.next_batch(batch_size=5000)
                x1_for_val, x2_for_val, y_tmp2 = balanced_batch(x1_in_val, y1_in_val)
                y_for_val = y_tmp2.reshape((-1, 1))

                _p, cur_loss = sess.run([pred, loss], feed_dict={x1_val: x1_for_val,
                                                                 x2_val: x2_for_val,
                                                                 y_val: y_for_val})

                print('epoch{%d}, acc: %lf, loss: %lf' % ((epoch + 1), get_acc(y_for_val, _p), cur_loss))
                print("ave_dis for pos: %lf; ave_dis for neg: %lf" % (np.average(_p[y_tmp2 == 0.]), np.average(_p[y_tmp2 == 1.])))


if __name__ == '__main__':
    work()
