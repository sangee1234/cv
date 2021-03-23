import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('test.csv')
print(train.columns)
print(train['filename'].unique())
print(train['cell_type'].value_counts())
