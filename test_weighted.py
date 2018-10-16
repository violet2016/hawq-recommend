import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.feature_column import _LazyBuilder
def test_weighted_categorical_column():
    color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}  # 4行样本
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_weight_categorical_column = feature_column.weighted_categorical_column(color_column, 'weight')
    builder = _LazyBuilder(color_data)
    with tf.Session() as session:
        id_tensor, weight = color_weight_categorical_column._get_sparse_tensors(builder)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('weighted categorical' + '-' * 40)
        print(session.run([id_tensor]))
        print('-' * 40)
        print(session.run([weight]))
test_weighted_categorical_column()