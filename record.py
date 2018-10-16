import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc
#https://jhui.github.io/2017/11/21/TensorFlow-Importing-data/
def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    #categorical_column = fc.categorical_column_with_hash_bucket(
   # key='plan_ops', hash_bucket_size=1000)
    #weighted_column = fc.weighted_categorical_column(
    #categorical_column=categorical_column, weight_feature_key='op_freq')
    #feature_env = fc.numeric_column('env', shape=(1,4), dtype=tf.int64)
    feature_label = fc.numeric_column('label', shape=(1,), dtype=tf.float32)
    feature_columns = [feature_label]
    fmap = fc.make_parse_example_spec(feature_columns)
    print(fmap)
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
    #print(features['plan_ops'])
    return env, reshape_label

EPOCHS = 10
BATCH_SIZE = 100
learning_rate = 0.01
# using two numpy arrays
#dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
dataset = tf.data.TFRecordDataset("./data/queries_samples/03")
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=1000, seed=10).repeat()

dataset = dataset.batch(BATCH_SIZE)


iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
#loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        
        # make a simple model

        #print("run", sess.run(next_ele))
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))