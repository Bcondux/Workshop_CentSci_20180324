import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

simlulated_sample_no = 500
Answer_W = np.array(np.random.random([5]))

noise_x = np.random.random([simlulated_sample_no, 5]) * 0.01
noise_y = np.random.random([simlulated_sample_no]) * 0.01
Observed_x = np.random.random([simlulated_sample_no, 5])
Observed_y = np.sum(Observed_x * Answer_W, axis = 1) 

# Observed_x += noise_x
Observed_y += noise_y
Observed_y = Observed_y.reshape([-1,1])

## Tensorflow part
x = tf.placeholder(tf.float32, [None,5])
y = tf.placeholder(tf.float32, [None,1])
W = tf.Variable(tf.random_normal([5]))
b = tf.Variable([0.0])
# fx = tf.reduce_sum(x * W, axis=1) + b
fx = tf.reduce_sum(x * W, axis=1) 
loss = tf.reduce_sum( tf.pow( (y-fx) , 2) * (1/simlulated_sample_no) )
# opt = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run(loss, feed_dict={x:Observed_x, y:Observed_y})
# sess.run(opt, feed_dict={x:Observed_x, y:Observed_y})
for i in range(50000):
    sess.run(opt, feed_dict={x:Observed_x, y:Observed_y})
    # sess.run(tf.reduce_sum(loss), feed_dict={x:Observed_x, y:Observed_y})
    if i % 100 == 0:
        Closs = sess.run(loss, feed_dict={x:Observed_x, y:Observed_y})
        [CW, CB] = sess.run([W, b])
        print(Answer_W, CW, CB, Closs)
    pass
pass
sess.close()
