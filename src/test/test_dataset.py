import tensorflow.compat.v1 as tf

filepath = '../dataset/test_data.csv'


def labels(features):
    labels = {}
    for label in ['click']:
        print(label)
        labels[label] = features.pop(label)
    return features, labels


data = tf.data.experimental.make_csv_dataset(file_pattern=[filepath], batch_size=64).map(labels)

for i in data:
    print(i)
    break
