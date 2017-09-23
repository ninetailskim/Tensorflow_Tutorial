'''
    Hello Tensorflow
'''

import tensorflow as tf

output = tf.constant('Hello Tensorflow')

with tf.Session() as sess:
    print(sess.run(output))