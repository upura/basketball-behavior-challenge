import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    X_train = Data.load('../input/X_train.pkl')
    y_train = Data.load('../input/y_train.pkl')
    X_test = Data.load('../input/X_test.pkl')

    nn_oof = pd.read_csv(f'../output/pred/oof_nn000.csv', header=None)
    nn_pred = pd.read_csv(f'../output/pred/pred_nn000.csv', header=None)

    X_train['nn'] = nn_oof.values
    X_test['nn'] = nn_pred.values

    fe_name = 'fe000'
    Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
