#--coding=utf-8---

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 64
mnist = input_data.read_data_sets('./data/mnist', one_hot=False)

print(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)

def single_model(X, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        w_1 = tf.get_variable('w1', initializer=tf.random_normal(shape=[784, 500]))
        b_1 = tf.get_variable('b1', initializer=tf.random_normal(shape=[500]))

        w_2 = tf.get_variable('w2', initializer=tf.random_normal(shape=[500, 10]))
        b_2 = tf.get_variable('b2', initializer=tf.random_normal(shape=[10]))

        m_1 = tf.matmul(X, w_1) + b_1
        r_1 = tf.nn.relu(m_1)

        m_2 = tf.matmul(r_1, w_2) + b_2
        r_2 = tf.nn.relu(m_2)
    return r_2


def dual_train_model(X1, X2, reuse_1, reuse_2):
    m1 = single_model(X1, reuse=reuse_1)
    m2 = single_model(X2, reuse=reuse_2)
    return m1, m2


def dual_test_model(X1, X2):
    m1 = single_model(X1, True)
    m2 = single_model(X2, True)
    return m1, m2


def train_graph(X1, X2, Y):
    y1, y2 = dual_train_model(X1, X2, False, True)
    with tf.name_scope('train'):
        E_w = tf.sqrt(tf.reduce_sum(tf.square(y1 - y2)))

        loss_1 = tf.multiply(tf.multiply(tf.subtract(tf.constant(1., dtype=tf.float32), Y),
                                         tf.constant(2. / 5., dtype=tf.float32)),
                             tf.square(E_w))
        loss_2 = tf.multiply(tf.multiply(Y, tf.constant(2. * 5., dtype=tf.float32)),
                             tf.exp(tf.multiply(tf.constant(-2.77 / 5., dtype=tf.float32), E_w)))
        loss = tf.reduce_sum(tf.add(loss_1, loss_2))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)
    return y1, y2, loss_1, loss_2, loss, train


def test_graph(X1, X2, Y):
    y1, y2 = dual_test_model(X1, X2)
    Y_pred = np.array(y1 != y2, dtype=np.float32)

    with tf.name_scope('test'):
        E_w = tf.sqrt(tf.reduce_sum(tf.square(y1 - y2)))

        loss_1 = tf.multiply(tf.multiply(tf.subtract(tf.constant(1., dtype=tf.float32), Y),
                                         tf.constant(2. / 5., dtype=tf.float32)),
                             tf.square(E_w))
        loss_2 = tf.multiply(tf.multiply(Y, tf.constant(2. * 5., dtype=tf.float32)),
                             tf.exp(tf.multiply(tf.constant(-2.77 / 5., dtype=tf.float32), E_w)))
        loss = tf.reduce_sum(tf.add(loss_1, loss_2))

        correct_prediction = tf.equal(Y, Y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    return accuracy, loss


def work():
    x1_train = tf.placeholder(tf.float32, shape=[None, 784], name='x1_train')
    x2_train = tf.placeholder(tf.float32, shape=[None, 784], name='x2_train')
    y_train = tf.placeholder(tf.float32, shape=[None, 1], name='y_train')
    #train = train_graph(x1_train, x2_train, y_train)
    y_1_test, y_2_test, loss_1_test, loss_2_test, loss_test, train = train_graph(x1_train, x2_train, y_train)


    x1_val = tf.placeholder(tf.float32, shape=[None, 784], name='x1_val')
    x2_val = tf.placeholder(tf.float32, shape=[None, 784], name='x2_val')
    y_val = tf.placeholder(tf.float32, shape=[None, 1], name='y_val')
    acc, loss = test_graph(x1_val, x2_val, y_val)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(10000):
            x1, y1 = mnist.train.next_batch(batch_size=batch_size)
            x2, y2 = mnist.train.next_batch(batch_size=batch_size)
            tmp = []
            for item in np.array(y1 != y2, dtype=np.float32):
                tmp.append([item])
            y = np.array(tmp, dtype=np.float32)

            sess.run([train], feed_dict={x1_train: x1,
                                         x2_train: x2,
                                         y_train: y})
            # test
            if epoch < 100:
                print('epoch ', epoch)
                a_1, a_2, a_3, a_4, a_5, a_6 = sess.run([y_1_test, y_2_test, loss_1_test, loss_2_test, loss_test, train],
                         feed_dict={x1_train: x1,
                                    x2_train: x2,
                                    y_train: y})

                print('loss_1', a_3)
                print('loss_2', a_4)
                print('loss', a_5)


            if epoch % 100 == 99:
                x1_in_val, y1_in_val = mnist.validation.next_batch(batch_size=50)
                x2_in_val, y2_in_val = mnist.validation.next_batch(batch_size=50)
                tmp_val = []
                for item in np.array(y1 != y2, dtype=np.float32):
                    tmp_val.append([item])
                y_in_val = np.array(tmp_val, dtype=np.float32)

                cur_acc, cur_loss = sess.run([acc, loss], feed_dict={x1_val: x1_in_val,
                                                                     x2_val: x2_in_val,
                                                                     y_val: y_in_val})
                print(cur_acc.shape, cur_loss.shape)
                print(cur_acc)
                print(cur_loss)
                print('epoch{%d}, acc: %lf, loss: %lf' % ((epoch + 1), cur_acc, cur_loss))


if __name__ == '__main__':
    work()


