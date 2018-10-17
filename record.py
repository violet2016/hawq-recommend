import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

categorical_column = fc.categorical_column_with_hash_bucket(key='query_plan_ops', hash_bucket_size=20)
weighted_column = fc.weighted_categorical_column(categorical_column=categorical_column, weight_feature_key='op_freq')
#feature_env = fc.numeric_column('env', shape=(1,4), dtype=tf.int64)
#feature_label = fc.numeric_column('label', shape=(1,), dtype=tf.float32)
#env_columns = tf.FixedLenFeature([1, 4], tf.int64)
#exec_time = tf.FixedLenFeature([], tf.float32)
env_columns = fc.numeric_column('env', (1, 4))
total_ops = fc.numeric_column('total_ops')
exec_time = fc.numeric_column('label')
feature_columns = [env_columns, weighted_column, total_ops, exec_time]

fmap = fc.make_parse_example_spec(feature_columns)
#fmap['env'] = env_columns
#fmap['label'] = exec_time
#print(fmap)
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
    label = tf.cast(features['label'], tf.float32)
    #ops = fc.embedding_column(features['query_plan_ops'], dimension=8) 
    ops = features['query_plan_ops']
    freq = features['op_freq']
    reshape_label = tf.reshape(features['label'], (1,1))
    #print(features['plan_ops'])
    return features['env'], ops, freq, reshape_label

EPOCHS = 100000
BATCH_SIZE = 100
learning_rate = 0.01
# using two numpy arrays
#dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
dataset = tf.data.TFRecordDataset("./data/queries_samples/08")
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=1000, seed=10).repeat()

dataset = dataset.batch(BATCH_SIZE)


iter = dataset.make_one_shot_iterator()
x, ops, freq, y = iter.get_next()
y_norm = tf.layers.batch_normalization(inputs=y,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training = True,
        name='y_norm')
transformed_columns = [
    env_columns,
    tf.feature_column.embedding_column(
        weighted_column, dimension=8)
]
inputs = tf.feature_column.input_layer({'env' :x, 'query_plan_ops' : ops, 'op_freq': freq}, transformed_columns)

net = tf.layers.dense(inputs, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
#prediction = tf.Print(prediction, [prediction], "Prediction: ")
#y_norm = tf.Print(y_norm, [y_norm], "Y is: ")
loss = tf.losses.mean_squared_error(prediction, tf.reshape(y_norm, (100,1))) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()

#loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        
        # make a simple model

        #print("run", sess.run(ops))
        _, loss_value = sess.run([train_op, loss])

        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
        if loss_value < 0.001:
            break
    save_path = saver.save(sess, "./data/models/simple_net.ckpt")
