import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc
dataset = tf.data.TFRecordDataset("./data/queries_samples/02")
iter = dataset.make_one_shot_iterator()
next_ele = iter.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        print("run", sess.run(next_ele))