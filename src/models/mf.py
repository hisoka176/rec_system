import tensorflow.compat.v1 as tf


def model_fn(features, labels, mode, params):
    main_input_layer = params['input_layer_mapping']

    inputs = tf.feature_column.input_layer(features, feature_columns=main_input_layer['main'])

    logits = tf.reshape(tf.layers.dense(inputs=inputs, units=1, activation=None), (-1,))
    output = tf.nn.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={
                'ctr': output
            }
        )
    labels = {}
    for label in params['dataset']['labels']:
        labels[label] = tf.cast(features.pop(label), tf.float32)

    predictions = tf.cast(tf.greater_equal(output, 0.5), tf.float64)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['click'], logits=logits))
    metric_auc = tf.metrics.auc(labels=labels['click'], predictions=output)
    metric_precision = tf.metrics.precision(labels=labels['click'], predictions=predictions)
    metric_recall = tf.metrics.precision(labels=labels['click'], predictions=predictions)
    metric_loss = tf.metrics.mean(output)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, eval_metric_ops={
                'auc': metric_auc,
                'precision': metric_precision,
                'recall': metric_recall,
                'metric_loss': metric_loss,
            }, loss=loss
        )
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['train']['lr'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    tensors_to_log = {
        "global_step": tf.train.get_global_step(),
        "train_loss": loss,
        'auc': metric_auc[0],
        'precision': metric_precision[0],
        'recall': metric_recall[0]
    }

    training_hooks = [tf.estimator.LoggingTensorHook(tensors=tensors_to_log,
                                                     every_n_iter=100)]
    output_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, training_hooks=training_hooks
    )
    return output_spec
