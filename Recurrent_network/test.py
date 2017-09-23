import tensorflow as tf
'''
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
c = tf.stack([a,b],axis=1)
d = tf.unstack(c,axis=0)
e = tf.unstack(c,axis=1)
f = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
g1 = tf.unstack(f,axis=0)
g2 = tf.unstack(f,axis=1)
#g3 = tf.unstack(f,axis=3)
h = tf.constant([[[1,2,1,2],[3,4,3,4]],
                [[5,6,5,6],[7,8,7,8]],
                [[9,0,9,0],[11,12,11,12]]])

i1 = tf.unstack(h,axis=0)
i2 = tf.unstack(h,axis=1)
i3 = tf.unstack(h,axis=2)
print(c.get_shape())
print(a.get_shape())
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(e[0]))
    print(sess.run(e[1]))
    print(sess.run(g1))
    print(sess.run(g2))
    #print(sess.run(g3))
    print(sess.run(i1))
    print(sess.run(i2))
    print(sess.run(i3))
    print(sess.run(i1[0]))
    print(sess.run(i2[0]))
    print(sess.run(i3[0]))
    print(i1[0].get_shape())
    print(i2[0].get_shape())
    print(i3[0].get_shape())
    
'''
'''
import tensorflow as tf
import numpy as np

cell = tf.contrib.rnn.BasicRNNCell(num_units=128)
print(cell.state_size)

inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = cell.zero_state(32, np.float32)
output, h1 = cell(inputs, h0)

print(h1.shape)
print(output.shape)
'''
'''
import tensorflow as tf
import numpy as np
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = lstm_cell.zero_state(32, np.float32)
output, h1 = lstm_cell(inputs, h0)

print(h1.h.shape)  
print(h1.c.shape)  
print(output.shape)
'''
'''
import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,0])

print(a)
print(a.shape)

b = tf.reshape(a, [1, 10])
print(b)
print(b.shape)
'''

for x in range(5):
    print(x)