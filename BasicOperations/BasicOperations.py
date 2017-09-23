'''
    Basic Operations
'''

import tensorflow as tf

const1 = tf.constant(5)
const2 = tf.constant(7)

var1 = tf.placeholder(tf.int32)
var2 = tf.placeholder(tf.int32)

matrix1 = tf.placeholder(tf.int32, shape=[2, 2])
matrix2 = tf.placeholder(tf.int32, shape=[2, 2])

add = tf.add(matrix1, matrix2)
multi = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    print('constant Addition %i' %sess.run(tf.add(const1, const2)))
    print('constant Multiplication %i' %sess.run(tf.multiply(const1, const2)))
    print('variable Addition %i' %sess.run(tf.add(var1, var2), feed_dict={var1:50, var2:70}))
    print('variable Multiplication %i' %sess.run(tf.multiply(var1, var2), feed_dict={var1:50, var2:70}))

    print(sess.run(add, feed_dict={matrix1:[[1, 2], [3, 4]], matrix2:[[2, 2], [2, 2]]}))
    print(sess.run(multi, feed_dict={matrix1:[[1, 2], [3, 4]], matrix2:[[2, 2], [2, 2]]}))

    print(sess.run(tf.add(matrix1, var1), feed_dict={matrix1:[[1, 2], [3, 4]], var1:20}))
    print(sess.run(tf.multiply(matrix1, var1), feed_dict={matrix1:[[1, 2], [3, 4]], var1:20}))

