'''

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32,[n_samples])
Y = tf.placeholder(tf.float32,[n_samples])

W = tf.get_variable("w", [1])
b = tf.get_variable("b", [1])

pred = tf.add(tf.multiply(X, W), b)
cost = tf.sqrt(tf.reduce_sum(tf.pow(pred - Y, 2)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer,cost], feed_dict={X:train_X, Y:train_Y})

        if epoch % display_step == 0:
            print("Epoch:", epoch , 'Cost:', c, 'W:', sess.run(W), 'b:', sess.run(b))
    
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()