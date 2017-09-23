'''
    one_hot = False
'''

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)

xdis, ydis = mnist.train.next_batch(5000)
xsam, ysam = mnist.train.next_batch(100)


xdistri = tf.placeholder(tf.float32, shape=[None,784])
xsample = tf.placeholder(tf.float32, shape=[784])

distance = tf.reduce_sum(tf.pow(tf.add(xdistri, tf.negative(xsample)), 2), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(xsam)):

        index = sess.run(pred, feed_dict={xdistri: xdis, xsample: xsam[i, :]})
        #print(index)
        #print('Sample', i, 'Prediction:', np.argmax(ydis[index]), 'True label:', np.argmax(ysam[i]))
        print('Sample', i, 'Prediction:', ydis[index], 'True label:', ysam[i])

        if ydis[index] == ysam[i]:
            accuracy += 1./len(xsam)
    
    print("Accuracy:", accuracy)