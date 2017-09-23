import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

X = tf.placeholder(tf.float32, [None, num_input])

weights = {
    'encoder_h1': tf.get_variable('W_e_1', [num_input, num_hidden_1], tf.float32, tf.contrib.layers.xavier_initializer()),
    'encoder_h2': tf.get_variable('W_e_2', [num_hidden_1, num_hidden_2], tf.float32, tf.contrib.layers.xavier_initializer()),
    'decoder_h1': tf.get_variable('W_d_1', [num_hidden_2, num_hidden_1], tf.float32, tf.contrib.layers.xavier_initializer()),
    'decoder_h2': tf.get_variable('W_d_2', [num_hidden_1, num_input], tf.float32, tf.contrib.layers.xavier_initializer()),
}
biases = {
    'encoder_h1': tf.get_variable('B_e_1', [num_hidden_1], tf.float32, tf.contrib.layers.xavier_initializer()),
    'encoder_h2': tf.get_variable('B_e_2', [num_hidden_2], tf.float32, tf.contrib.layers.xavier_initializer()),
    'decoder_h1': tf.get_variable('B_d_1', [num_hidden_1], tf.float32, tf.contrib.layers.xavier_initializer()),
    'decoder_h2': tf.get_variable('B_d_2', [num_input], tf.float32, tf.contrib.layers.xavier_initializer()),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_h2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_h2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        _, l = sess.run([optimizer, loss], feed_dict={X:batch_x})

        if i % display_step == 0:
            print('loss', l)
        
    n = 4
    batch_x, _ = mnist.test.next_batch(1)
    g = sess.run(decoder_op, feed_dict={X:batch_x})
    ori = batch_x.reshape([28, 28])
    res = g.reshape([28, 28])

    plt.figure()
    plt.imshow(ori)
    plt.show()

    plt.figure()
    plt.imshow(res)
    plt.show()