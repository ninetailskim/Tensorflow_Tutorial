'''

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'layer1': tf.get_variable('layer1_w', [n_input, n_hidden_1]),
    'layer2': tf.get_variable('layer2_w', [n_hidden_1, n_hidden_2]),
    'out': tf.get_variable('out_w', [n_hidden_2, n_classes])
}

biases = {
    'layer1': tf.get_variable('layer1_b', [n_hidden_1]),
    'layer2': tf.get_variable('layer2_b', [n_hidden_2]),
    'out': tf.get_variable('out_b', [n_classes]) 
}

def struct(x):
    layer_1 = tf.add(tf.matmul(x, weights['layer1']), biases['layer1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['layer2']), biases['layer2'])
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer


logits = struct(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        total_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={X:batch_x, Y:batch_y})

            total_cost += c

        if epoch % display_step == 0:
            print("Epoch:", epoch, "cost:", total_cost / total_batch)
    print("Optimization finished!")

    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))