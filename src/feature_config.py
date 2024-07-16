import tensorflow.compat.v1 as tf


def str2dtype(dtype):
    res = tf.int64
    if dtype == 'string':
        res = tf.string
    elif dtype == 'int':
        res = tf.int64
    elif dtype == 'float':
        res = tf.float64
    else:
        raise NotImplementedError('dtype not implement')
    return res


def build_input_layer(input_layer, feature_column_mapping):
    input_layer_mapping = {}
    print(feature_column_mapping.keys())
    for input_layer_name, input_layer_features in input_layer.items():
        input_layer_mapping[input_layer_name] = [feature_column_mapping[name] for name in input_layer_features]
    return input_layer_mapping


def build_feature_column(features):
    mapping = {}
    for feature in features:
        key = feature['name']
        if feature.get('mode', '') == 'category':
            fc = tf.feature_column.categorical_column_with_hash_bucket(key=key, hash_bucket_size=feature['bucket'],
                                                                       dtype=tf.int64)
            ec = tf.feature_column.embedding_column(fc, dimension=feature['dim'])
            mapping[key] = ec
    return mapping
