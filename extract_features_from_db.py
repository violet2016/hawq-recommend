import psycopg2
import sys
import time
import csv
import tensorflow as tf
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
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

hostname = 'localhost'
username = 'vcheng'
database = 'hawq-recommend'

def get_query_samples(conn, to_file, from_config):
    cur = conn.cursor()

    cur.execute("select  query_plan_rows, cpu_size, mem_size, storage_size, data_size,\
                query_plan_op_list, \
                table_size_list , cpu_percent_list, total_exec_time_in_ms\
                from samples where config_id >= %s and total_exec_time_in_ms is not NULL" % from_config)
    feature_data = {}
    writer = tf.python_io.TFRecordWriter(to_file)
   # with open(to_file, mode='w') as dst_file:
        #dst_writer = csv.writer(dst_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in cur.fetchall():
            # feature_data = {
            #     'query_plan_row_length': row[0],
            #     'env_segment_number': len(row[7]),
            #     'env_segment_cpu': row[1],
            #     'env_segment_mem': row[2],
            #     'env_segment_storage': row[3],
            #     'query_plan_ops': [str.encode(my_str) for my_str in row[5]],
            #     'query_table_size': row[6],
            #     'segment_cpu_usage' : ['{0:.4g}'.format(num) for num in row[7]],
            #     'exec_time': row[8]
            # }
        if row[8] == None or len(row[5]) == 0:
            print("time is one or no plan, skip")
            continue
        all_ops = {}
        total = 0
        for op in row[5]:
            total = total + 1
            if op in all_ops:
                all_ops[op] += 1
            else:
                all_ops[op] = 1
        feature_data = {
            'env': [row[0], len(row[7]), row[1], row[2]],
            'query_plan_ops': [str.encode(d) for d in all_ops.keys()], #[str.encode(my_str) for my_str in row[5]],
            'op_freq': [d/total for d in all_ops.values()],
            'total_ops': total,
            'label': row[8]
        }
        print(feature_data)
        #dst_writer.writerow(row[0], len(row[7]), row[1], row[2], row[8])
        
        # write label, shape, and image content to the TFRecord file
        example = tf.train.Example(features=tf.train.Features(feature={
                    'env': _floatlist_feature(feature_data['env']),
                    'query_plan_ops': _byteslist_feature(feature_data['query_plan_ops']),
                    'op_freq': _floatlist_feature(feature_data['op_freq']),
                    'total_ops': _float_feature(feature_data['total_ops']),
                    'label': _float_feature(feature_data['label'])
                    }))
        writer.write(example.SerializeToString())
    writer.close()
filename = sys.argv[1]
fromconfigid = sys.argv[2]
myConnection = psycopg2.connect( host=hostname, user=username, dbname=database )
get_query_samples( myConnection, './data/queries_samples/'+filename, fromconfigid)
myConnection.close()
