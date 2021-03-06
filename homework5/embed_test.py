#--coding=utf-8---

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR =  "minimalsample"
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 640


path_for_mnist_sprites =  os.path.join(LOG_DIR,'mnistdigits.png')
path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')


def pad_data(x):
    x = np.reshape(x, (-1, 28, 28, 1))
    return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
x_batch, y_batch = mnist.train.next_batch(TO_EMBED_COUNT)
batch_xs = pad_data(x_batch)
y_l = np.argmax(y_batch, axis=1)
batch_ys = np.array(y_l).reshape([-1, 1])
#print(y_batch)





def embeding_config():
    embedding_var = tf.Variable(np.zeros([TO_EMBED_COUNT, 84]), name=NAME_TO_VISUALISE_VARIABLE, dtype=tf.float32)

    summary_writer = tf.summary.FileWriter(LOG_DIR)
    summary_writer.add_graph(tf.get_default_graph())

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = path_for_mnist_metadata  # 'metadata.tsv'

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = path_for_mnist_sprites  # 'mnistdigits.png'
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)
    return embedding_var


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits


def sprite_and_meta_writer(batch_xs, batch_ys):
    to_visualise = batch_xs
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    with open(path_for_mnist_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batch_ys):
            f.write("%d\t%d\n" % (index, label))


if __name__ == "__main__":

    saver = tf.train.import_meta_graph('./save/lenet_save-5000.meta')

    fc2_samples = np.zeros((TO_EMBED_COUNT, 84))

    with tf.Session() as sess:
        saver.restore(sess, './save/lenet_save-5000')

        X = sess.graph.get_tensor_by_name('x:0')
        Y = sess.graph.get_tensor_by_name('y:0')

        fc2 = sess.graph.get_tensor_by_name('LeNet_Var/LeNet/fc_2/Relu:0')

        fc2_samples = sess.run([fc2], feed_dict={X: batch_xs, Y: y_batch})[0]

        embedding_var = embeding_config()
        sess.run(tf.variables_initializer([embedding_var]))

        print(sess.run(embedding_var))
        sess.run(tf.assign(embedding_var, fc2_samples))
        print(sess.run(embedding_var))


        saver_p = tf.train.Saver()
        saver_p.save(sess, os.path.join(LOG_DIR, "embeding_model.ckpt"), 1)
        sprite_and_meta_writer(x_batch, batch_ys)

