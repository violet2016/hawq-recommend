import unittest
import os
from feature import *

testfile = 'data/test.tfrecord'
class feature_test(unittest.TestCase):
    def test_write_and_read(self):
        feature_data = {
            'query_plan_row_length': 17,
            'env_segment_number': 7,
            'env_segment_cpu': 1000,
            'env_segment_mem': 500,
            'env_segment_storage': 2000,
            'query_plan_ops': [b'Gather Motion', b'Sort', b'HashAggregate', b'Redistribute Motion', b'HashAggregate', b'Parquet table Scan'],
            'query_table_size': [5.99861e+07],
        }
        write_to_tfrecord(feature_data, testfile)
        filename_queue = tf.train.string_input_producer([ testfile ])
        context_parsed, table_size, query_plan_ops = read_and_decode(filename_queue, 200)
        print(table_size)

if __name__ == '__main__':
    os.remove(testfile)
    unittest.main()