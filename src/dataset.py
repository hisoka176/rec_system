import argparse
import os

import pandas as pd


def main(args):
    pd.set_option('display.max_columns', None)
    filepath = args.input
    data = pd.read_csv(filepath)

    # data['user_id'] = data['user_id'].apply('str')
    data['user_id'] = data['user_id'].astype('str')
    # data['item_id'] = data['item_id'].apply('str')
    data['item_id'] = data['item_id'].astype('str')
    train_data = data.sample(frac=0.8, axis=0)
    other_data = data[~data.index.isin(train_data.index)]
    dev_data = other_data.sample(frac=0.5, axis=0)
    test_data = other_data[~other_data.isin(dev_data.index)]

    train_data.to_csv(os.path.join(args.output, 'train_data.csv'), index=False)
    dev_data.to_csv(os.path.join(args.output, 'dev_data.csv'), index=False)
    test_data.to_csv(os.path.join(args.output, 'test_data.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='input')
    parser.add_argument('--output', type=str, default='/kaggle/input', help='output')
    args = parser.parse_args()
    main(args)
