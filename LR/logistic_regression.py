'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import pandas as pd
import numpy as np
import tensorflow as tf
def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'env': tf.FixedLenFeature([1, 4], tf.int64),
            # 'env_segment_number': tf.FixedLenFeature([], tf.int64),
            # 'env_segment_cpu': tf.FixedLenFeature([], tf.int64),
            # 'env_segment_mem': tf.FixedLenFeature([], tf.int64),
            # 'query_plan_ops': tf.VarLenFeature(tf.string),
            # 'query_table_size': tf.VarLenFeature(tf.float32),
            # 'segment_cpu_usage': tf.VarLenFeature(tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
        })
    env = tf.cast(features['env'], tf.float32)
    # image.set_shape([DEPTH * HEIGHT * WIDTH])

    # # Reshape from [depth * height * width] to [depth, height, width].
    # image = tf.cast(
    #     tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
    #     tf.float32)
    label = tf.cast(features['label'], tf.float32)
    reshape_label = tf.reshape(features['label'], (1,1))
    return env, reshape_label

EPOCHS = 10
BATCH_SIZE = 100

# Parameters
learning_rate = 0.01
training_epochs = 25
display_step = 1


# tf Graph Input
#x = tf.placeholder(tf.float32, [4, 1]) # mnist data image of shape 28*28=784
#y = tf.placeholder(tf.float32, [1, None])

# Set model weights
W = tf.Variable(tf.zeros([4, 1]))
b = tf.Variable(tf.zeros([1]))

# Gradient Descent


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
handle = tf.placeholder(tf.string, shape=[])

# Start training
with tf.Session() as sess:
    #filename_queue = tf.train.string_input_producer(['./data/queries_samples/01'], num_epochs=1)
    # Run the initializer
    #cols = ['seg','cpu','mem','label']

    #train = pd.read_csv('../data/queries_samples/01.csv',delimiter=',',names = cols)
    #test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)
    record_defaults = [tf.float32] * 4   # Eight required float columns
    dataset = tf.data.TFRecordDataset("../data/queries_samples/02").repeat()
    dataset = dataset.map(parser)
    #dataset = dataset.batch(BATCH_SIZE)
    iter = dataset.make_one_shot_iterator()
    sess.run(init)
    x, y = iter.get_next()
    #env, label = read_and_decode(filename_queue)
    #dataset = dataset.repeat(training_epochs)

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        #batch = dataset.batch(batch_size)
        total_batch = 100
        
        #train_iterator_handle = sess.run(iter.string_handle())
        avg_cost = 0.
        # Loop over all batches
        for i in range(EPOCHS):
            #batch_xs, batch_ys = dataset.batch(batch_size)
            
            #next_element = iter.get_next()
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run([x, y])
            _, c = sess.run([optimizer, cost])
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            # Compute average loss
            #avg_cost += c / total_batch
        # Display logs per epoch step
        
            

    print("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
