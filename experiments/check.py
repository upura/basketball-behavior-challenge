import pandas as pd

cat = 'train'
sid = '0000'
X_seq = pd.read_csv(f'../input/{cat}/{sid}.csv')

print(X_seq.columns)
print(X_seq.head())

print((X_seq['scr_x'].diff() ** 2 + X_seq['scr_y'].diff() ** 2) ** (1 / 2))
