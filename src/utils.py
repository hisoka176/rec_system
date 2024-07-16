import importlib


def load_module(module):
    name = f'models.{module}'
    print(f'load_module ==== {name}')
    module = importlib.import_module(name=name)
    return module.model_fn


if __name__ == '__main__':
    d = load_module('mf')
    print(d)
