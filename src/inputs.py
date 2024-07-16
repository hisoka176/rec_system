import tensorflow.compat.v1 as tf


def input_fn(mode, params):
    file_pattern = params['dataset']['train_dataset'] if mode == tf.estimator.ModeKeys.TRAIN else params['dataset'][
        'test_dataset']
    dataset = tf.data.experimental.make_csv_dataset(file_pattern=file_pattern, header=True, batch_size=2048,
                                                    prefetch_buffer_size=4096)
    if params['train']['epoch'] > 1:
        dataset = dataset.repeat(params['train']['epoch'] - 1)
    elements = dataset.make_one_shot_iterator().get_next()
    return elements, None
