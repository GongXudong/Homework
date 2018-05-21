#--coding=utf-8---
import tensorflow as tf

class LeNet:
    def __init__(self, x, y, mu, sigma):
        self.x = x
        self.y = y
        self.mu = mu
        self.sigma = sigma

        tf.set_random_seed(1)
        with tf.variable_scope('LeNet_Var') as scope:
            self.logits = self.net_build()
            scope.reuse_variables()
            self.logits_val = self.net_build()

        #train part
        self.pred = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), 1))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        #validation part
        self.pred_val = tf.nn.softmax(self.logits_val)
        correct_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred_val, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_train_op(self):
        return self.train_op

    def get_loss_and_acc(self):
        return self.loss_val, self.acc

    def cnnLayer(self, x, kHeight, kWidth, strideX, strideY, featureNum, name, padding='SAME'):
        channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('w', shape=[kHeight, kWidth, channels, featureNum], initializer=tf.truncated_normal_initializer(self.mu, self.sigma), dtype=tf.float32)
            b = tf.get_variable('b', shape=[featureNum], initializer=tf.zeros_initializer(tf.float32))
            featureMap = tf.nn.conv2d(x, w, strides=[1, strideX, strideY, 1], padding=padding)
            out = tf.nn.bias_add(featureMap, b)
            return tf.nn.relu(out, name=scope.name)
            # return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)

    def maxPoolLayer(self, x, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1], strides=[1, strideX, strideY, 1], padding=padding, name=name)

    def fcLayer(self, x, inputD, outputD, reluFlag, name):
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('w', shape=[inputD, outputD], dtype=tf.float32)
            b = tf.get_variable('b', shape=[outputD], dtype=tf.float32)
            out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
            if reluFlag:
                return tf.nn.relu(out)
            else:
                return out

    def net_build(self, reuse=False):
        with tf.variable_scope('LeNet') as scope:
            conv_1 = self.cnnLayer(self.x, 5, 5, 1, 1, 6, 'conv_1', padding='VALID')
            pool_1 = self.maxPoolLayer(conv_1, 2, 2, 2, 2, 'maxpool_1', padding='VALID')

            conv_2 = self.cnnLayer(pool_1, 5, 5, 1, 1, 16, 'conv_2', padding='VALID')
            pool_2 = self.maxPoolLayer(conv_2, 2, 2, 2, 2, 'maxpool_2', padding='VALID')

            fc0 = tf.contrib.layers.flatten(pool_2)

            fc1 = self.fcLayer(fc0, 400, 120, reluFlag=True, name='fc_1')
            fc2 = self.fcLayer(fc1, 120, 84, reluFlag=True, name='fc_2')
            logits = self.fcLayer(fc2, 84, 10, reluFlag=False, name='logits')

        return logits

