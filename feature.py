import tensorflow as tf


def read_and_decode(filename_queue, num_classes):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        'query_plan_row_length': tf.FixedLenFeature([], tf.int64),
        'env_segment_number': tf.FixedLenFeature([], tf.int64),
        'env_segment_cpu': tf.FixedLenFeature([], tf.int64),
        'env_segment_mem': tf.FixedLenFeature([], tf.int64),
        'env_segment_storage': tf.FixedLenFeature([], tf.int64)
    }
    
    sequence_features = {
        'query_plan_ops': tf.VarLenFeature(tf.string),
    #    'query_plan_ops_seq': tf.VarLenFeature(tf.int64),
        'query_table_size': tf.VarLenFeature(tf.float32)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    # Defaults are not specified since both keys are required.
    )

    query_plan_ops = tf.sparse_to_indicator(sequence_parsed['query_plan_ops'], num_classes)
    query_plan_ops.set_shape([None, num_classes])
    return context_parsed, sequence_features['query_table_size'], query_plan_ops

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _byteslist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_to_tfrecord(feature_data_map, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
                'query_plan_row_length': _int64_feature(feature_data_map['query_plan_row_length']),
                'env_segment_number': _int64_feature(feature_data_map['env_segment_number']),
                'env_segment_cpu': _int64_feature(feature_data_map['env_segment_cpu']),
                'env_segment_mem': _int64_feature(feature_data_map['env_segment_mem']),
                'env_segment_storage': _int64_feature(feature_data_map['env_segment_storage']),
                'query_plan_ops': _byteslist_feature(feature_data_map['query_plan_ops']),
                'query_table_size': _floatlist_feature(feature_data_map['query_table_size']),
                }))
    writer.write(example.SerializeToString())
    writer.close()

# def get_all_records(FILE):
#     with tf.Session() as sess:
#         filename_queue = tf.train.string_input_producer([ FILE ])
#         context, table_size, plan_ops = read_and_decode(filename_queue)
#         #image = tf.reshape(image, tf.pack([height, width, 3]))
#         #image.set_shape([720,720,3])
#         init_op = tf.initialize_all_variables()
#         sess.run(init_op)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         for i in range(2053):
#             example, l = sess.run([image, label])
#             print (example,l)
#         coord.request_stop()
#         coord.join(threads)

# get_all_records('/path/to/train-0.tfrecords')