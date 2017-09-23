'''
    cell.__call__()
'''
import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

X = tf.placeholder(tf.float32, [None, timesteps, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


weights={
    'out':tf.get_variable('weight_out', [num_hidden, num_classes], tf.float32)
}

biases = {
    'out':tf.get_variable('biases_out', [num_classes], tf.float32)
}
    

def RNN(x, weight, biases):
    x = tf.unstack(x, timesteps, 1)
    
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
    h0 = lstm_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope("test") as scope:
        for i in range(timesteps - 1):
            if i > 0:
                scope.reuse_variables()
            output, h0 = lstm_cell(x[i], h0)
    out = tf.nn.bias_add(tf.matmul(output, weight['out']), biases['out']) 
        
    return out 


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        _, loss, acc = sess.run([train_op, loss_op, accuracy],feed_dict={X:batch_x, Y:batch_y})

        if step % display_step == 0:
            print("Step:", step, "loss:", loss, "Accuracy:", acc)
    print("Training finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Test:", sess.run(accuracy, feed_dict={X:test_data, Y:test_label}))