import pandas as pd

cat = 'train'
sid = '0000'
X_seq = pd.read_csv(f'../input/{cat}/{sid}_feat.csv')

print(X_seq)
print(X_seq.std(axis=0).values)
print(X_seq / X_seq.std(axis=0).values)
