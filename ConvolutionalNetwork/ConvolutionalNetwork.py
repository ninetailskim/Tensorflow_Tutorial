'''

'''

import tensorflow as tf
import tools

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)


learning_rate = 0.001
num_steps = 2000
batch_size = 128

num_input = 784
num_classes = 10
dropout = 0.75
is_pretrain = False

def Network(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x['images']
        x = tf.reshape(x, [-1,28,28,1])
        x = tools.conv('conv1', x, 32, kernel_size=[5, 5], is_pretrain=is_pretrain)
        x = tools.pool('pool1', x, is_max_pool=True)
        x = tools.conv('conv2', x, 64, is_pretrain=is_pretrain)
        x = tools.pool('pool2', x, is_max_pool=True)

        fc_x = tools.FC_layer('FC1', x, 1024)

        fc_x = tf.layers.dropout(fc_x, rate=dropout, training=is_training)

        out = tools.FC_layer('FC2', fc_x, n_classes)

    return out

def model_fn(features, labels, mode):
    logits_train = Network(features, num_classes, dropout, reuse=False, is_training=True)

    logits_test = Network(features, num_classes, dropout, reuse=True, is_training=False)
    #test
    pred_classes = tf.argmax(logits_test, axis=1)
    #test
    pred_probs = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, prediction=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op=tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op}
    )
    return estim_specs


model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
    )

model.train(input_fn,steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=False
)

e = model.evaluate(input_fn)

print('Test Accuracy:', e['accuracy'])