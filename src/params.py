import json

from config import load_yaml, build_flags
from feature_config import build_feature_column, build_input_layer


def initialize():
    params = load_yaml('resource/config.yaml')
    print(json.dumps(params, indent=4))
    feature_column_mapping = build_feature_column(params['feature']['config'])
    input_layer_mapping = build_input_layer(params['feature']['input_layer'], feature_column_mapping)
    print('==== feature_column ====')
    for index, (fc_name, fc) in enumerate(feature_column_mapping.items()):
        print(f'index:{index}:feature_name:{fc_name}==>{fc}')
    print('==== input_layer ====')
    for index, (il_name, il) in enumerate(input_layer_mapping.items()):
        print(f'index:{index}:layer_main: {il_name}')
        for il_fc in il:
            print(f'\t\tfeature_column:{il_fc}')
        # for il_feature in il:
        #     print(f'\t\t{il_feature}')

    params['feature_column_mapping'] = feature_column_mapping
    params['input_layer_mapping'] = input_layer_mapping

    FLAGS = build_flags()
    params['flags'] = FLAGS

    return params


params = initialize()
