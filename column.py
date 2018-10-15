#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from builtins import str

import tensorflow as tf
from tensorflow import feature_column as fc

_SUPPORT_CLASSIFIER_ = ['lr', 'dnn', None]


def _verify_column_args(name, classifier):
    if not isinstance(name, str):
        raise TypeError("{}: name must be string,"
                        "while get {}".format(name, type(name)))

    if classifier not in _SUPPORT_CLASSIFIER_:
        raise ValueError(
            "{}: classifier should be in {}, "
            "while get: {}".format(name, _SUPPORT_CLASSIFIER_, classifier))


class Column(object):

    """数据列抽象类

    数据中各列的抽象类
    """

    def __init__(self, name, classifier):
        """初始化Column类

        :param name: str, 列名
        :param classifier: str, 选择分类器使用lr, dnn, None
        """

        # TODO: 专门校验函数，更好
        if classifier:
            classifier = classifier.lower()
        else:
            tf.logging.info(
                "column {} is not used, "
                "as its classifier is {}".format(name, classifier))
        _verify_column_args(name, classifier)

        self.name = name
        self.classifier = classifier

    def build(self, *args, **kwargs):
        """为生成feature_column需做的构建工作"""
        # TODO: 更合适的共用参数
        return self

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def get_feature_column(self):
        """用于tensorflow的FeatureColumn"""
        raise NotImplementedError()

    def get_feature_engineering_fn(self):
        """用于feature_engineering_fn函数"""
        raise NotImplementedError()

    def get_feature_spec(self):
        """用于serving_input_fn函数"""
        raise NotImplementedError()


def _gen_normalizer_fn(name, mean, std):
    mean = mean if mean else None

    if std:
        assert std > 0, (
            "{}: std > 0, while get {}".format(name, std))

    if mean and std:
        # TODO: Tensorflow可能有归一函数
        return lambda x: tf.divide(tf.subtract(x, mean), std)
    elif mean or std:
        tf.logging.warn("{}: mean and std are not used,"
                        "as they should be both specified,"
                        "while mean: {}, std: {}".format(name, mean, std))


class ContinuousColumn(Column):
    def __init__(self, name, classifier=None, **kwargs):
        """连续数值列

        例子：点击率
        :param name:
        :param kwargs:
        """
        super(ContinuousColumn, self).__init__(name, classifier)

        # TODO: 增加bucketized_column
        mean = kwargs.pop("mean", None)
        std = kwargs.pop("std", None)
        self._normalizer_fn = _gen_normalizer_fn(name, mean, std)
        self._feature_column = None

    def get_feature_column(self):
        if not self._feature_column:
            self._feature_column = fc.numeric_column(
                self.name,
                normalizer_fn=self._normalizer_fn)

        return self._feature_column

    def get_feature_engineering_fn(self):
        return None

    def get_feature_spec(self):
        return fc.make_parse_example_spec([self.get_feature_column()])


def _get_category_column_from_dict(name, kwargs):
    """生成对应的category feature column.

    :param name: 列名
    :param kwargs:
        下面四个参数互斥:
        * num_buckets: int, 最大数值
        * hash_buckets: int, hash桶数
        * vocab_list: list, 候选词列表
        * vocab_file: str, 候选词文件
    :return: feature_column
    """
    convert_methods = {"num_buckets", "hash_buckets",
                       "vocab_list", "vocab_file"}
    convert_method = set(kwargs.keys()).intersection(convert_methods)

    # 四个参数互斥
    if len(convert_method) > 1:
        raise ValueError("{}: {} cannot coexist."
                         "Please leave only one".format(name, convert_method))
    elif len(convert_method) < 1:
        raise ValueError(
            "{}: argument missing,"
            "one of {} should be given".format(name, convert_methods))
    else:
        convert_method = list(convert_method)[0]

    if convert_method == "num_buckets":
        # TODO: 检查dtype == int
        num_buckets = kwargs.pop(convert_method)
        assert num_buckets > 0
        # TODO: 如果能用-1更好
        # missing_value用最大值填充
        default_value = num_buckets
        num_buckets += 1

        return fc.categorical_column_with_identity(
            name, num_buckets, default_value)
    elif convert_method == "hash_buckets":
        hash_bucket_size = kwargs.pop(convert_method)
        assert hash_bucket_size > 0
        # TODO: 支持dtype传入
        return fc.categorical_column_with_hash_bucket(
            name, hash_bucket_size)
    elif convert_method == "vocab_list":
        vocabulary_list = kwargs.pop(convert_method)
        if isinstance(vocabulary_list, str):
            # TODO: 词表支持不同的分隔符
            vocabulary_list = [x.strip() for x in vocabulary_list.split(",")]

        assert len(vocabulary_list) > 1, (
            "{}: list's length > 0,"
            "while get: {}".format(name, len(vocabulary_list)))

        # TODO: 支持dtype, default_value
        return fc.categorical_column_with_vocabulary_list(
            name, vocabulary_list)
    elif convert_method == "vocab_file":
        # TODO: 支持文件
        raise NotImplementedError("Support later.")
    else:
        raise ValueError("{}: {} is not supported".format(
            name, convert_method))


def _create_category_output_column(name, input_column, output_type, **kwargs):
    """生成category相应的输出FeatureColumn.

    :param name: str, 列名
    :param input_column: list, category feature column
    :param output_type:
    :param kwargs:
    :return:
    """
    if output_type == "embeding":
        assert 'dimension' in kwargs, "{}: missing dimension".format(name)
        dimension = kwargs["dimension"]
        assert dimension > 0, "dimension > 0"

        return fc.embedding_column(input_column, dimension)
    elif output_type == "indicator":
        return fc.indicator_column(input_column)
    else:
        raise ValueError(
            "{}: {} is not supported,"
            "use embeding or indicator either".format(name, output_type))


class CategoricalColumn(Column):
    # TODO: 完善文档中参数说明
    def __init__(self, name, classifier=None,
                 output_type="embeding", **kwargs):
        """单值离散列

        例子：性别
        :param name:
        :param output_type:
        :param kwargs:
        """
        super(CategoricalColumn, self).__init__(name, classifier)

        self._input_column = _get_category_column_from_dict(
            self.name, kwargs)
        self._feature_column = _create_category_output_column(
            self.name, self._input_column, output_type, **kwargs)

        self._unused_args = kwargs

    def get_input_column(self):
        return self._input_column

    def get_feature_column(self):
        return self._feature_column

    def get_feature_engineering_fn(self):
        return None

    def get_feature_spec(self):
        return fc.make_parse_example_spec([self.get_feature_column()])


class MultitalentCategoricalColumn(CategoricalColumn):
    # TODO: 完善文档中参数说明
    def __init__(self, name, classifier=None,
                 delimiter=",", output_type="embeding", **kwargs):
        """多值离散特征（文本）

        例子：标签
        :param name:
        :param delimiter: char, 数据文本按此分割
        :param output_type:
        :param kwargs:
        """
        super(MultitalentCategoricalColumn, self).__init__(
            name, classifier, output_type, **kwargs)

        assert isinstance(delimiter, str) and len(delimiter) == 1, (
            "delimiter must be a char, while get: {}".format(delimiter))
        self.delimiter = delimiter

    def get_feature_engineering_fn(self):
        def fn(features, labels):
            features[self.name] = tf.string_split(features[self.name],
                                                  delimiter=self.delimiter)
            return features, labels

        return fn

    def get_feature_spec(self):
        return {
            self.name: tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=[""])
        }


def _split_tag_tensor(str_tensor, delimiter, sep):
    terms = tf.string_split(str_tensor, delimiter + sep)

    tags = tf.sparse_tensor_to_dense(terms, default_value="")
    tag_names = tags[:, 0::2]

    # TODO: 转换两次，性能差
    tags = tf.sparse_tensor_to_dense(terms, default_value="0")
    tag_weights = tf.string_to_number(tags[:, 1::2])

    return tag_names, tag_weights


class TagColumn(Column):
    def __init__(self, name, classifier=None,
                 delimiter="|", sep="@", **kwargs):
        """微博标签类

        :param name:
        :param index:
        :param dtype:
        :param delimiter: char, 标签间的分隔符，默认是@
        :param sep: char, 标签内，名字和权重的分隔符，默认是|
        :param kwargs:
        """
        super(TagColumn, self).__init__(name, classifier)

        assert isinstance(delimiter, str) and len(delimiter) == 1, (
            "{}: delimiter must be char,"
            "while get: {}".format(name, delimiter))
        self.delimiter = delimiter

        assert isinstance(sep, str) and len(sep) == 1, (
            "{}: sep must be char, while get: {}".format(name, sep))
        self.sep = sep

        self._tag_name = name + "_tag"
        self._weight_name = name + "_weight"

        if "num_buckets" in kwargs:
            raise ValueError("num_buckets is not supported in TagColumn.")
        self._input_column = _get_category_column_from_dict(
            self._tag_name, kwargs)
        self._weight_input_column = fc.weighted_categorical_column(
            self._input_column, self._weight_name)
        self._feature_column = fc.indicator_column(self._weight_input_column)

    def get_feature_engineering_fn(self):
        # TODO: 变成全局函数
        def fn(features, labels):
            tag, weight = _split_tag_tensor(features[self.name],
                                            self.delimiter, self.sep)
            features[self._tag_name] = tag
            features[self._weight_name] = weight
            return features, labels

        return fn

    def get_feature_column(self):
        # TODO: 统一将_feature_column上移到父类Column
        return self._feature_column

    def get_feature_spec(self):
        return {
            self.name: tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=[""])
        }


def _split_string_to_fix_words(line, delimiter, max_words):
    words = tf.string_split(line, delimiter)

    fix_shape = [words.dense_shape[0], max_words]
    fix_words = tf.sparse_reset_shape(
        tf.sparse_slice(words, [0, 0], fix_shape),
        fix_shape)
    return fix_words


def _build_conv_layer_fn_from_text(name, delimiter,
                                   max_vocabs_in_sentence,
                                   vocab_size,
                                   dimension, conv_size):
    def fn(features, labels):
        words = _split_string_to_fix_words(features[name],
                                           delimiter,
                                           max_vocabs_in_sentence)

        def hash(x):
            return tf.string_to_hash_bucket_fast(x, vocab_size)

        ids = tf.SparseTensor(words.indices,
                              tf.map_fn(hash, words.values, dtype=tf.int64),
                              words.dense_shape)
        # 固定词汇数，如果超过则截断，不足则补vocab_size（最大值）
        # TODO: 最好用-1来补，但embed_sequence似乎不支持
        ids = tf.sparse_tensor_to_dense(ids, default_value=vocab_size)

        embs = tf.contrib.layers.embed_sequence(
            ids,
            vocab_size=vocab_size + 1,
            embed_dim=dimension)
        embs_expand = tf.expand_dims(embs, axis=3)

        conv = tf.contrib.layers.conv2d(
            embs_expand, num_outputs=dimension,
            kernel_size=[conv_size, dimension],
            padding="VALID")

        pool_size = max_vocabs_in_sentence - conv_size + 1
        pool = tf.contrib.layers.max_pool2d(
            conv, kernel_size=[pool_size, 1])

        features[name] = tf.reshape(pool, [-1, dimension])
        return features, labels

    return fn


class TextColumn(Column):
    def __init__(self, name,
                 max_vocabs_in_sentence, vocab_size, dimension, conv_size,
                 classifier=None, delimiter=" ", **kwargs):
        super(TextColumn, self).__init__(name, classifier)

        # TODO: 统一函数来验证，代码冗余
        assert isinstance(max_vocabs_in_sentence, int), (
            "max_vocabs_in_sentence must be int,"
            "while get: {}".format(type(max_vocabs_in_sentence)))
        self.max_vocabs_in_sentence = max_vocabs_in_sentence

        assert isinstance(vocab_size, int), (
            "vocab_size must be int, while get: {}".format(type(vocab_size)))
        self.vocab_size = vocab_size

        assert isinstance(dimension, int), (
            "dimension must be int, while get: {}".format(type(dimension)))
        self.dimension = dimension

        assert isinstance(conv_size, int) and conv_size < vocab_size, (
            "conv_size < vocab_size, while get: {}".format(conv_size))
        self.conv_size = conv_size

        assert isinstance(delimiter, str) and len(delimiter) == 1, (
            "delimiter must be char, while get: ".format(delimiter))
        self.delimiter = delimiter

        self._feature_column = None

    def get_feature_engineering_fn(self):
        return _build_conv_layer_fn_from_text(
            self.name, self.delimiter,
            self.max_vocabs_in_sentence,
            self.vocab_size, self.dimension, self.conv_size)

    def get_feature_column(self):
        if not self._feature_column:
            self._feature_column = fc.numeric_column(
                self.name, shape=(1, self.dimension))

        return self._feature_column

    def get_feature_spec(self):
        return {
            self.name: tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=[""])
        }


class CrossColumn(Column):
    def __init__(self, name, keys, hash_bucket_size,
                 classifier="lr", **kwargs):
        super(CrossColumn, self).__init__(name, classifier)
        assert self.classifier != "dnn", (
            "CrossColumn doesn't supports dnn classifier now")

        assert isinstance(keys, str), ("keys must be string,"
                                       "while get {}".format(keys))
        keys = [x.strip() for x in keys.split(",")]
        assert len(keys) >= 2, ("cross need >= 2 keys,"
                                "while get {}".format(keys))
        self.key_names = keys
        self.keys = None

        assert hash_bucket_size > 0, ("hash_bucket_size > 0,"
                                      "while get {}".format(hash_bucket_size))

        self.hash_bucket_size = hash_bucket_size

        self._feature_column = None

    def build(self, columns_dict):
        """ 组建crossed_column

        :param columns_dict: 已组建的类表{name: columns}
        :return:
        """

        if not self.keys:
            columns = [columns_dict[k] for k in self.key_names]
            invalid_cols = [f for f in columns
                            if not isinstance(f, CategoricalColumn)]
            if invalid_cols:
                raise ValueError(
                    "{} are not CategoricalColumn".format(invalid_cols))
            self.keys = [c.get_input_column() for c in columns]

        if not self._feature_column:
            self._feature_column = fc.crossed_column(self.keys,
                                                     self.hash_bucket_size)
        return self

    def get_feature_column(self):
        assert self._feature_column is not None, (
            "cross column is not initialize,"
            "use self.__call__ to build it first")
        return self._feature_column

    def get_feature_engineering_fn(self):
        return None


def make_serving_input_spec(columns):
    # TODO: 更高效的方式
    spec = {}
    for c in columns:
        spec.update(c.get_feature_spec())
    return spec


def make_feature_engineering_fn(columns):
    fns_names = [(c.get_feature_engineering_fn(), c.name) for c in columns]
    fns_names = [(fn, name) for (fn, name) in fns_names if fn]

    if fns_names:
        def all_fn(features, labels):
            with tf.name_scope("columns"):
                for (fn, name) in fns_names:
                    with tf.name_scope(name):
                        features, labels = fn(features, labels)
                return features, labels

        return all_fn
    else:
        return None
