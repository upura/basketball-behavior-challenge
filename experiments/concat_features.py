import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    X_train = Data.load('../input/X_train_fe002.pkl')
    y_train = Data.load('../input/y_train_fe002.pkl')
    X_test = Data.load('../input/X_test_fe002.pkl')

    nn_oof = pd.read_csv('../output/pred/oof_nn000.csv', header=None)
    nn_pred = pd.read_csv('../output/pred/pred_nn000.csv', header=None)

    X_train['nn'] = nn_oof.values
    X_test['nn'] = nn_pred.values

    fe_name = 'fe004'
    Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
