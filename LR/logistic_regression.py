'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import pandas as pd
from __future__ import print_function
import numpy as np
import tensorflow as tf

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
dataset = tf.data.TFRecordDataset("../data/queries_samples/01")
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 50
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [1, 4]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [1, None])

# Set model weights
W = tf.Variable(tf.zeros([4, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
handle = tf.placeholder(tf.string, shape=[])

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    obj = tf.parse_single_example(
        serialized_example,
        features={
            'env': tf.FixedLenFeature([1, 4], tf.float32),
            # 'env_segment_number': tf.FixedLenFeature([], tf.int64),
            # 'env_segment_cpu': tf.FixedLenFeature([], tf.int64),
            # 'env_segment_mem': tf.FixedLenFeature([], tf.int64),
            # 'query_plan_ops': tf.VarLenFeature(tf.string),
            # 'query_table_size': tf.VarLenFeature(tf.float32),
            # 'segment_cpu_usage': tf.VarLenFeature(tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
      }
    )
   # plan_metrics_box = tf.stack([tf.strings.to_number(obj['query_%s' % x].values, tf.int64)
    #                         for x in ['plan_ops', 'table_size']])
    env = obj['env']
    label = obj['label']#, np.std(np.array(obj['segment_cpu_usage'])), 100-np.average(np.array(obj['segment_cpu_usage'])))
    return env, label

# Start training
with tf.Session() as sess:
    #filename_queue = tf.train.string_input_producer(['./data/queries_samples/01'], num_epochs=1)
    # Run the initializer
    cols = ['seg','cpu','mem','label']

    train = pd.read_csv('../data/queries_samples/01.csv',delimiter=',',names = cols)
    #test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)

    sess.run(init)
    #env, label = read_and_decode(filename_queue)
    #dataset = dataset.repeat(training_epochs)
    
    # Training cycle
    for epoch in range(training_epochs):
        batch = dataset.batch(batch_size)
        iterator = batch.make_one_shot_iterator()
        train_iterator_handle = sess.run(iterator.string_handle())
        avg_cost = 0.
        # Loop over all batches
        while True:
            next_element = iterator.get_next()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: env, y: label})
            # Compute average loss
            #avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
