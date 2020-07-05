import numpy as np
import pandas as pd
import torch


class BasketDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_train = False if self.y is None else True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sid = str(i).zfill(4)
        cat = 'train' if self.is_train else 'test'
        X_seq_pad = np.zeros(shape=(242, 6), dtype=np.float)
        X_seq = pd.read_csv(f'../input/{cat}/{sid}_feat.csv')
        X_seq_pad[:len(X_seq), :] = X_seq
        return (
            torch.tensor(X_seq_pad).float(),
            torch.tensor(self.X.iloc[i].values).float(),
            torch.tensor(self.y.iloc[i]).float()
        ) if self.is_train else (
            torch.tensor(X_seq_pad).float(),
            torch.tensor(self.X.iloc[i].values).float()
        )
