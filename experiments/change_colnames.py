import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    X_train = Data.load(f'../input/X_train_fe002.pkl')
    X_test = Data.load(f'../input/X_test_fe002.pkl')

    colnames = X_train.columns
    colnames = [c.replace('(2, 5, 10, 20)', '_2_5_10_20_').replace('"', '_') for c in colnames]
    pd.Series(colnames).to_csv('../input/colnames.csv', index=False)

    X_train.columns = colnames
    X_test.columns = colnames
    Data.dump(X_train, f'../input/X_train_fe002.pkl')
    Data.dump(X_test, f'../input/X_test_fe002.pkl')
