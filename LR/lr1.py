import tensorflow as tf
from numpy import *

x_train = [[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 5.0], [1.0, 3.0], [4.0, 2.0], [7.0, 3.0], [4.0, 5.0], [11.0, 3.0],
           [8.0, 7.0]]
y_train = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_train = mat(y_train)

theta = tf.Variable(tf.zeros([2, 1]))
theta0 = tf.Variable(tf.zeros([1, 1]))
y = 1 / (1 + tf.exp(-tf.matmul(x_train, theta) + theta0))

loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(1000):
    sess.run(train)
print(step, sess.run(theta).flatten(), sess.run(theta0).flatten())