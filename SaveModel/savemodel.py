'''

'''

import tensorflow as tf
import tools

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


learning_rate = 0.001
num_steps = 2000
batch_size = 128
num_epochs = 200

num_display = 2000

num_input = 784
num_classes = 10
dropout = 0.75
is_pretrain = False
model_path = '/tmp/model.ckpt'


x = tf.placeholder(tf.float32, [None, num_input])
y = tf.placeholder(tf.float16, [None, num_classes])
is_training= True

def Network(x, n_classes, dropout):
    global is_training
    x = tf.reshape(x, [-1,28,28,1])
    x = tools.conv('conv1', x, 32, kernel_size=[5, 5], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, is_max_pool=True)
    x = tools.conv('conv2', x, 64, is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, is_max_pool=True)

    fc_x = tools.FC_layer('FC1', x, 1024)

    fc_x = tf.layers.dropout(fc_x, rate=dropout, training=is_training)

    out = tools.FC_layer('FC2', fc_x, n_classes)

    return out


pred = Network(x, num_classes, dropout)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

accuracy = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
acc = tf.reduce_mean(tf.cast(accuracy, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    is_training=True
    for epoch in range(num_epochs):
        total_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, loss], feed_dict={x:batch_x, y:batch_y})
            total_cost += c
            
            if i % num_display == 0:
                print('loss', total_cost / total_batch)    
        

    print('Training Finished')

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)
    is_training = False
    print('accuracy:', sess.run(acc, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    
    