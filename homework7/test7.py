import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import utils
from tensorflow.examples.tutorials.mnist import input_data
import os

BATCH_SIZE=10

def pad_data(x):
    x = np.reshape(x, (-1, 28, 28, 1))
    return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
batch_img, batch_label = mnist.test.next_batch(BATCH_SIZE)
batch_img = pad_data(batch_img)
batch_size = BATCH_SIZE



if __name__ == "__main__":

    saver = tf.train.import_meta_graph('./save/lenet_save-5000.meta')


    # with eval_graph.as_default():
    #
    #     with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
    #         x = tf.placeholder(tf.float32, [None, 32, 32, 1], name='x')
    #         y_ = tf.placeholder(tf.float32, [None, 10], name='y')
    #         lenet = LeNet(x, y_, 0., 0.01)
    #
    #         cost = (-1) * tf.reduce_sum(tf.multiply(y_, tf.log(tf.nn.softmax(tf.))), axis=1)
    #
    #         # Guided backpropagtion back to input layer
    #
    #         gb_grad = tf.gradients(cost, x)[0]
    #
    #         # gradient for partial linearization. We only care about target visualization class.
    #
    #         y_c = tf.reduce_sum(tf.multiply(lenet.logits, labels), axis=1)
    #
    #         print('y_c:', y_c)
    #
    #         # Get last convolutional layer gradient for generating gradCAM visualization
    #
    #         target_conv_layer = lenet.pool2
    #
    #         target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
    #
    #


    with tf.Session() as sess:

        saver.restore(sess, './save/lenet_save-5000')

        images = sess.graph.get_tensor_by_name('x:0')
        labels = sess.graph.get_tensor_by_name('y:0')

        pool2 = sess.graph.get_tensor_by_name("LeNet_Var/LeNet/maxpool_2:0")

        logits = sess.graph.get_tensor_by_name("LeNet_Var/LeNet/logits/LeNet_Var/LeNet/logits:0")

        predict = tf.nn.softmax(logits=logits)

        prob = sess.run(predict, feed_dict={images: batch_img})


        cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(predict)), axis=1)
        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, images)[0]

        # gradient for partial linearization. We only care about target visualization class.
        y_c = tf.reduce_sum(tf.multiply(logits, labels), axis=1)
        print('y_c:', y_c)
        # Get last convolutional layer gradient for generating gradCAM visualization
        target_conv_layer = pool2
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

        gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
        [gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: batch_label})


        for i in range(batch_size):
            #utils.print_prob(prob[i], './synset.txt')

            grad = gb_grad_value[i]

            utils.visualize(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], grad, size=32)


