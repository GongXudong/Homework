import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    A = tf.placeholder(tf.float32, [3, 4])
    B = tf.placeholder(tf.float32, [4, 3])
    C = tf.placeholder(tf.float32, [3, 3])

    res = tf.matmul(A, B) + C

    with tf.Session() as sess:

        np.random.seed(1)

        randomA = np.random.random(size=[3, 4])
        randomB = np.random.random(size=[4, 3])
        randomC = np.random.random(size=[3, 3])

        print("Matrix A:")
        print(randomA)
        print("Matrix B:")
        print(randomB)
        print("Matrix C:")
        print(randomC)

        print("result from numpy:")
        print(np.dot(randomA, randomB) + randomC)

        resRes = sess.run([res], feed_dict={A: randomA,
                                            B: randomB,
                                            C: randomC})
        print("result from tensorflow:")
        print(resRes)

