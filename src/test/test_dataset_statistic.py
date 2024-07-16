import pandas as pd
import tensorflow.compat.v1 as tf

filepath = '../dataset/test_data.csv'

df = pd.read_csv(filepath)
# 统计一下分桶的个数，了解一下
print(df.user_id.nunique())
print(df.item_id.nunique())
#
