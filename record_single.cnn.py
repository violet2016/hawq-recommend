import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 6, 6, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

categorical_column = fc.categorical_column_with_hash_bucket(key='query_plan_ops', hash_bucket_size=200)
weighted_column = fc.weighted_categorical_column(categorical_column=categorical_column, weight_feature_key='op_freq')
#feature_env = fc.numeric_column('env', shape=(1,4), dtype=tf.int64)
#feature_label = fc.numeric_column('label', shape=(1,), dtype=tf.float32)
#env_columns = tf.FixedLenFeature([1, 4], tf.int64)
#exec_time = tf.FixedLenFeature([], tf.float32)
cpu_column = fc.numeric_column('cpu', (1,1))
env_columns = fc.numeric_column('env', (1, 3))
total_ops = fc.numeric_column('total_ops')
exec_time = fc.numeric_column('label')
cat_table_size = fc.categorical_column_with_hash_bucket(key='table_size', hash_bucket_size=20)
weighted_column_table = fc.weighted_categorical_column(categorical_column=cat_table_size, weight_feature_key='table_size_weight')
feature_columns = [ cpu_column, env_columns, weighted_column, total_ops, exec_time, weighted_column_table]

fmap = fc.make_parse_example_spec(feature_columns)
#fmap['env'] = env_columns

#fmap['label'] = exec_time
print(fmap)
#https://jhui.github.io/2017/11/21/TensorFlow-Importing-data/
def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    

    features = tf.parse_single_example(
        serialized_example,
        # features={
        #     'env': tf.FixedLenFeature([1, 4], tf.int64),
        #     # 'env_segment_number': tf.FixedLenFeature([], tf.int64),
        #     # 'env_segment_cpu': tf.FixedLenFeature([], tf.int64),
        #     # 'env_segment_mem': tf.FixedLenFeature([], tf.int64),
        #     #'query_plan_ops': tf.VarLenFeature(tf.string),
        #     #'query_table_size': tf.VarLenFeature(tf.float32),
        #     #'segment_cpu_usage': tf.VarLenFeature(tf.float32),
        #     'label': tf.FixedLenFeature([], tf.float32)
        # }
        fmap
        )
    #env = tf.cast(features['env'], tf.float32)

    # image.set_shape([DEPTH * HEIGHT * WIDTH])

    # # Reshape from [depth * height * width] to [depth, height, width].
    # image = tf.cast(
    #     tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
    #     tf.float32)
    label = tf.cast(features['label'], tf.int32)
    #ops = fc.embedding_column(features['query_plan_ops'], dimension=8) 
    ops = features['query_plan_ops']
    freq = features['op_freq']
    reshape_label = tf.one_hot(label, depth=5)
    reshape_cpu = tf.reshape(features['cpu'], (1,1))
    size_weight = features['table_size_weight']
    ts_string = tf.cast(features['table_size'], tf.string)
    #print("size is", size_weight, "string is ", ts_string)
    return reshape_cpu, features['env'], ts_string, size_weight , ops, freq, reshape_label

EPOCHS = 1000
BATCH_SIZE = 1
learning_rate = 0.01
# using two numpy arrays
#dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
dataset = tf.data.TFRecordDataset("./data/queries_samples/14")
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=1000, seed=10).repeat()

dataset = dataset.batch(BATCH_SIZE)

#test_data = (np.array([[1,2]]), np.array([[0]]))

#iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#iter = iterator.make_initializer(dataset)
#iter = dataset.make_initializable_iterator()
iter = dataset.make_one_shot_iterator()
c, e, table_string, table_size, ops, freq, y = iter.get_next()
# y_norm = tf.layers.batch_normalization(inputs=y,
#         axis=-1,
#         momentum=0.99,
#         epsilon=0.01,
#         center=True,
#         scale=True,
#         training = True,
#         name='y_norm')
# y_norm = tf.Print(y_norm, [y_norm], "y norm is")
transformed_columns = [
    cpu_column,
    tf.feature_column.embedding_column(
        weighted_column, dimension=20),
     tf.feature_column.embedding_column(
       weighted_column_table , dimension=12),
    env_columns
    
]
#x = tf.Print(e, [e, c, y], "x")
inputs = tf.feature_column.input_layer({'cpu': c, 'env': e, 'query_plan_ops' : ops, 'op_freq': freq, 'table_size': table_string, 'table_size_weight': table_size}, transformed_columns)
num_classes = 5

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([6, 6, 1, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([6, 6, 16, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([2*2*32, 64])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([64, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Construct model

logits = conv_net(inputs, weights, biases, keep_prob)
#logits = tf.Print(logits, [logits], "logits")
prediction = tf.nn.softmax(logits)
#prediction = tf.Print(prediction, [prediction], "prediction")
# Define loss and optimizer
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#   logits=logits, labels=y))
loss_op = tf.losses.mean_squared_error(prediction, tf.reshape(y, (1,5)))
#loss_op = tf.Print(loss_op, [loss_op], "loss_op")
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
#tf.add_to_collection('pred_network', prediction)
# one_hot_y = tf.one_hot(tf.cast(y, tf.int32), 1)
# loss = tf.losses.mean_squared_error(prediction, one_hot_y) # pass the second value 
# train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()


#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    #sess.run(iter.initializer)
    fail =0 
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        
        # make a simple model

        y_out = sess.run(y)
        print("y", y_out)
        loss_value, in_, pred= sess.run([loss_op,inputs, prediction], feed_dict={keep_prob: 0.8})

        
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
        print("pred", pred)


    save_path = saver.save(sess, "./data/models/simple_net.ckpt_14")
    # Test model
    #correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_norm, 1))
    # Calculate accuracy
    #correct_prediction = tf.Print(correct_prediction, [correct_prediction], "Correct Prediction")
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", sess.run(accuracy))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))