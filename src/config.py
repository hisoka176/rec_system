import json

import yaml
import os.path
import tensorflow.compat.v1 as tf

root_path = os.path.dirname(__file__)


def include(self, node):
    filename = os.path.join(root_path, f'resource/{node.value}')
    if os.path.exists(filename):
        with open(filename, 'r') as fr:
            return yaml.load(fr, Loader=yaml.FullLoader)


yaml.add_constructor('!include', include)


def load_yaml(file_name):
    """Load YAML file to be dict"""
    config_filepath = os.path.join(root_path, file_name)
    if os.path.exists(config_filepath):
        with open(file_name, 'r', encoding="utf-8") as fr:
            dict_obj = yaml.load(fr, Loader=yaml.FullLoader)
        return dict_obj
    else:
        raise FileNotFoundError(f'NOT Found YAML file {file_name} === {config_filepath}')


def build_flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('model', default='mf', help='model')
    return FLAGS


if __name__ == '__main__':
    yaml_dict = load_yaml("resource/config.yaml")
    print(json.dumps(yaml_dict, indent=4))
    FLAGS = build_flags()
    print(FLAGS.model)
