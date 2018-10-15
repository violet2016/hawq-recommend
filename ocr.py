# coding=utf-8
# 定义模型结构, 数据预处理.
import logging

import numpy as np
import tensorflow as tf

import re
import os
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 10, """dev的测试的batch size""")

tf.app.flags.DEFINE_string('data_dir', './data/queries_samples',"训练数据文件夹")
tf.app.flags.DEFINE_string('train_file', '01',"训练数据文件夹")
#tf.app.flags.DEFINE_string('train_file', 'train.record.0',"训练数据文件夹")
tf.app.flags.DEFINE_string('cv_file', 'cv.tfrecord',"训练数据文件夹")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0004,"lr")



def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    obj = tf.parse_single_example(
        serialized_example,
        features={
            'query_plan_row_length': tf.FixedLenFeature([], tf.int64),
            'env_segment_number': tf.FixedLenFeature([], tf.int64),
            'env_segment_cpu': tf.FixedLenFeature([], tf.int64),
            'env_segment_mem': tf.FixedLenFeature([], tf.int64),
            'query_plan_ops': tf.VarLenFeature(tf.string),
            'query_table_size': tf.VarLenFeature(tf.float64),
            'segment_cpu_usage': tf.VarLenFeature(tf.float64),
            'exec_time': tf.FixedLenFeature([], tf.float64)
      }
    )
    plan_metrics_box = tf.stack([obj['query_%s' % x].values
                             for x in ['ops', 'table_size']])
    env = tf.stack([obj['query_plan_row_length'], obj['env_segment_number'], obj['env_segment_cpu'], obj['env_segment_mem']])
    label = tf.multiply(obj['exec_time'], np.std(obj['segment_cpu_usage']), 100-np.average(obj['segment_cpu_usage']))

    return env, plan_metrics_box, label



def inference(feature_area, seq_len, batch_size):
  num_hidden = 320
  num_layers = 5
  num_classes = 10926 + 2

  with tf.device('/cpu:0'):
    cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, use_peepholes=True,
                                      initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                                      state_is_tuple=True)

    cells_fw = [cell_fw] * num_layers
    cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, use_peepholes=True,
                                      initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.1),
                                      state_is_tuple=True)

    cells_bw = [cell_bw] * num_layers

    w = tf.get_variable("weights", [num_hidden * 2, num_classes],
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=0.1))
    b = tf.get_variable("biases", [num_classes],
                        initializer=tf.constant_initializer(0.0))

  outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                 cells_bw,
                                                                 feature_area,
                                                               dtype=tf.float32,
                                                        sequence_length=seq_len)

  # 做一个全连接映射到label_num个数的输出
  outputs = tf.reshape(outputs, [-1, num_hidden * 2])
  logits = tf.add(tf.matmul(outputs, w), b, name="logits_add")
  logits = tf.reshape(logits, [batch_size, -1, num_classes])
  ctc_input = tf.transpose(logits, (1, 0, 2))
  return ctc_input



def inputs(filename, batch_size):
  
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename])

    env, plan_metrics_box, label = read_and_decode(filename_queue)

    env, plan_metrics_box, labels = tf.train.shuffle_batch(
        [env, plan_metrics_box, label], batch_size=batch_size, num_threads=2,
        capacity=10 + 3 * batch_size,
        min_after_dequeue=10)

    return  env, plan_metrics_box, labels

def run_training():
  batch_size = FLAGS.batch_size

  initial_learning_rate = 0.0001
  with tf.Session() as sess:
    global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False, dtype=tf.int32)

    train_filename = os.path.join(FLAGS.data_dir, FLAGS.train_file)
    #cv_filename = os.path.join(FLAGS.data_dir, FLAGS.cv_file)
    env, plan_metrics_box, labels = inputs(train_filename, batch_size=batch_size)
    #cv_images, cv_labels = inputs(cv_filename, batch_size=batch_size)

    with tf.variable_scope("inference") as scope:
      train_ctc_in = inference(images, [730]* batch_size, batch_size)
      scope.reuse_variables()
      dev_ctc_in = inference(cv_images, [730]* batch_size, batch_size)
        
    train_ctc_losses = tf.nn.ctc_loss( labels, train_ctc_in, [730] * batch_size)
    train_cost = tf.reduce_mean(train_ctc_losses, name="train_cost")
    # 限制梯度范围
    optimizer = tf.train.AdamOptimizer(initial_learning_rate)
    grads_and_vars = optimizer.compute_gradients(train_cost)
    capped_grads_and_vars = grads_and_vars
    capped_grads_and_vars = [(tf.clip_by_value(gv[0], -50.0, 50.0), gv[1]) for gv in grads_and_vars]

    train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                       global_step=global_step)


    # cv
    dev_decoded, dev_log_prob = tf.nn.ctc_greedy_decoder(dev_ctc_in, [730] * batch_size)
    dev_edit_distance = tf.edit_distance(tf.to_int32(dev_decoded[0]), cv_labels,
                                 normalize=False)
    dev_batch_error_count = tf.reduce_sum(dev_edit_distance)
    dev_batch_label_count = tf.shape(cv_labels.values)[0]



    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        sess.run(train_op)
        #_, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        if step % 100 == 0:
          dev_error_count = 0
          dev_label_count = 0

          for batch in range(10):
            dev_error_count_value, dev_label_count_value = sess.run(
              [dev_batch_error_count, dev_batch_label_count])
            dev_error_count += dev_error_count_value
            dev_label_count += dev_label_count_value

          dev_acc_ratio = (dev_label_count - dev_error_count) / dev_label_count
          print("eval_acc = %.3f " % dev_acc_ratio)
        step += 1
    except tf.errors.OutOfRangeError:
      pass
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()

if __name__ == "__main__":
  pass
  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  run_training()