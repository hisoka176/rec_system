"""
todo
 1. 热启动
 2. 循环学习率
 3. 保留验证集指标最高的那个
 4. 提前结束， 避免过拟合
 5. 多任务如何
 6. 所有的参数都要处理好
 7 esmm2的处理
 8. 时长的处理，使用ndcg处理
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)
from inputs import input_fn
from utils import load_module
from params import params

if tf.gfile.Exists('model_dir'):
    tf.gfile.DeleteRecursively('model_dir')

config = tf.estimator.RunConfig(
    model_dir='model_dir',
    keep_checkpoint_max=1
)

model_fn = load_module(params['flags'].model)

print('number', int(
    params['train']['train_dataset_size'] / params['train']['batch_size'] *
    params['train']['epoch']))
print('number 2', int(params['train']['dev_dataset_size'] / params['train']['batch_size']))
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)

loss_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator=estimator, metric_name='loss',
                                                               max_steps_without_decrease=100, run_every_secs=None,
                                                               run_every_steps=100)
auc_hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator=estimator, metric_name='auc',
                                                              max_steps_without_increase=100, run_every_secs=None,
                                                              run_every_steps=100)
hooks = [loss_hook, auc_hook]
# train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN, params=params),
#                                     max_steps=int(
#                                         params['train']['train_dataset_size'] / params['train']['batch_size'] *
#                                         params['train']['epoch']), hooks=hooks)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN, params=params),
                                    max_steps=None, hooks=hooks)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.EVAL, params=params),
                                  steps=int(params['train']['dev_dataset_size'] / params['train']['batch_size']),
                                  name='eval', hooks=[], )
tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
# print(result[0])
# print(result[1])
# with tf.gfile.Open('model_dir/metrics.json', 'w') as f:
#     print(metrics)
# f.write(str(metrics))
