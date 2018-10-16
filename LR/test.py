import tensorflow as tf
import os
learning_rate = 0.01
training_epochs = 25
batch_size = 50
display_step = 1
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/../data/queries_samples/01.csv"
features = tf.placeholder(tf.float32, [1, 4], name='features') # mnist data image of shape 28*28=784
time = tf.placeholder(tf.float32, name='time')

# Set model weights
W = tf.Variable(tf.zeros([4, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.nn.softmax(tf.matmul(features, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(time*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0], [0], [0], [0], [0.0]]
    country, code, gold, silver, bronze, total, time = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([gold, silver, bronze])
    return features, country

#features = tf.placeholder(tf.int32, shape=[4], name='features')
#time = tf.placeholder(tf.float32, name='time')
#total = tf.reduce_sum(features, name='total')
#printerop = tf.Print(time, [features], name='printer')
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            row, seg, cpu, mem, time_value = line.strip().split(",")
            row = int(row)
            cpu = int(cpu)
            mem = int(mem)
            time_value = float(time_value)
            # Run the Print ob
            _, c = sess.run([optimizer, cost], feed_dict={features: [[row, seg, cpu, mem]], time:time_value})
            print("cost=", "{:.9f}".format(c))

            #print(row, total)