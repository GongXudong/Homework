import tensorflow as tf
import numpy as np

a = tf.Variable(np.random.randint(0, 10))
b = tf.Variable(np.random.randint(0, 10))
c = tf.Variable(np.random.randint(0, 10))


tf.add_to_collection('init', a)
tf.add_to_collection('init', b)


with tf.Session() as sess:
    init_1 = tf.variables_initializer(tf.get_collection('init'))
    sess.run([init_1])

    try:
        res_c = sess.run([c])
    except tf.errors.FailedPreconditionError:
        sess.run(c.initializer)
    finally:
        print(sess.run([c]))

