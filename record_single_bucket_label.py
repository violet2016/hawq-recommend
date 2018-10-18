import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

categorical_column = fc.categorical_column_with_hash_bucket(key='query_plan_ops', hash_bucket_size=200)
weighted_column = fc.weighted_categorical_column(categorical_column=categorical_column, weight_feature_key='op_freq')
#feature_env = fc.numeric_column('env', shape=(1,4), dtype=tf.int64)
#feature_label = fc.numeric_column('label', shape=(1,), dtype=tf.float32)
#env_columns = tf.FixedLenFeature([1, 4], tf.int64)
#exec_time = tf.FixedLenFeature([], tf.float32)
cpu_column = fc.numeric_column('cpu', (1,1))
env_columns = fc.numeric_column('env', (1, 3))
total_ops = fc.numeric_column('total_ops')
exec_time = fc.bucketized_column('label')
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
    label = tf.cast(features['label'], tf.float32)
    #ops = fc.embedding_column(features['query_plan_ops'], dimension=8) 
    ops = features['query_plan_ops']
    freq = features['op_freq']
    reshape_label = tf.reshape(features['label'], (1,1))
    reshape_cpu = tf.reshape(features['cpu'], (1,1))
    size_weight = features['table_size_weight']
    ts_string = tf.cast(features['table_size'], tf.string)
    #print("size is", size_weight, "string is ", ts_string)
    return reshape_cpu, features['env'], ts_string, size_weight , ops, freq, reshape_label

EPOCHS = 10000
BATCH_SIZE = 1
learning_rate = 0.01
# using two numpy arrays
#dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
dataset = tf.data.TFRecordDataset("./data/queries_samples/12")
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=1000, seed=10).repeat()

dataset = dataset.batch(BATCH_SIZE)

#test_data = (np.array([[1,2]]), np.array([[0]]))

#iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#iter = iterator.make_initializer(dataset)
#iter = dataset.make_initializable_iterator()
iter = dataset.make_one_shot_iterator()
c, e, table_string, table_size, ops, freq, y = iter.get_next()
#y = tf.Print(y, [y], "y is")
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
        weighted_column, dimension=16),
     tf.feature_column.embedding_column(
       weighted_column_table , dimension=8),
    env_columns
    
]
x = tf.Print(e, [e, c, y], "x")
inputs = tf.feature_column.input_layer({'cpu': c, 'env': e, 'query_plan_ops' : ops, 'op_freq': freq, 'table_size': table_string, 'table_size_weight': table_size}, transformed_columns)

net = tf.layers.dense(inputs, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
tf.add_to_collection('pred_network', prediction)
#prediction = tf.Print(prediction, [prediction], "Prediction: ")
#y_norm = tf.Print(y_norm, [y_norm], "Y is: ")
loss = tf.multiply(1000., tf.losses.mean_squared_error(prediction, tf.reshape(y, (1,1)))) # pass the second value from iter.get_net() as label
#loss = tf.losses.mean_squared_error(tf.mean_squared_error(y-prediction), reduction_indices=1))
train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()


#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    #sess.run(iter.initializer)
    fail =0 
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        
        # make a simple model

        #print("run", sess.run(ops))
        _, loss_value, in_, pred = sess.run([train_op, loss,inputs, prediction])

        #print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
        if i > EPOCHS*0.75:
            print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
            #print(in_)
            fail = fail+1
        #if loss_value < 0.0001:
        #    break
    print("Fail", 4*fail/EPOCHS)
    save_path = saver.save(sess, "./data/models/simple_net.ckpt_another_loss")
    # Test model
    #correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_norm, 1))
    # Calculate accuracy
    #correct_prediction = tf.Print(correct_prediction, [correct_prediction], "Correct Prediction")
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", sess.run(accuracy))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))