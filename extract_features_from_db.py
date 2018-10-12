import psycopg2
import sys
from feature import *
import time
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
    take_a_rest_count = 0
    for row in cur.fetchall():
        take_a_rest_count = take_a_rest_count + 1
        feature_data = {
            'query_plan_row_length': row[0],
            'env_segment_number': len(row[7]),
            'env_segment_cpu': row[1],
            'env_segment_mem': row[2],
            'env_segment_storage': row[3],
            'query_plan_ops': [str.encode(my_str) for my_str in row[5]],
            'query_table_size': row[6],
            'segment_cpu_usage' : ['{0:.4g}'.format(num) for num in row[7]],
            'exec_time': row[8]
        }
        print(feature_data)
        write_to_tfrecord(feature_data, to_file)
filename = sys.argv[1]
fromconfigid = sys.argv[2]
myConnection = psycopg2.connect( host=hostname, user=username, dbname=database )
get_query_samples( myConnection, './data/queries_samples/'+filename, fromconfigid)
myConnection.close()
