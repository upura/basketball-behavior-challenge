import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features

from ayniy.utils import Data


def extract_feat(sid: str, cat: str = 'train'):
    df = pd.read_csv(f'../input/{cat}/{sid}.csv')

    df['column_id'] = sid
    df['diff_usr_ball'] = ((df['usr_x'] - df['bal_x'])**2 + (df['usr_y'] - df['bal_y'])**2)
    df['diff_usr_uDF'] = ((df['usr_x'] - df['uDF_x'])**2 + (df['usr_y'] - df['uDF_y'])**2)
    df['diff_usr_scr'] = ((df['usr_x'] - df['scr_x'])**2 + (df['usr_y'] - df['scr_y'])**2)
    df['diff_scr_uDF'] = ((df['scr_x'] - df['uDF_x'])**2 + (df['scr_y'] - df['uDF_y'])**2)
    df['diff_scr_ball'] = ((df['scr_x'] - df['bal_x'])**2 + (df['scr_y'] - df['bal_y'])**2)
    df['diff_uDF_ball'] = ((df['uDF_x'] - df['bal_x'])**2 + (df['uDF_y'] - df['bal_y'])**2)

    ef = extract_features(df, column_id='column_id')
    return ef


if __name__ == '__main__':

    X_train = []
    X_test = []

    for i in tqdm(range(1528)):
        X_train.append(extract_feat(sid=str(i).zfill(4), cat='train'))
    for i in tqdm(range(382)):
        X_test.append(extract_feat(sid=str(i).zfill(4), cat='test'))

    X_train = pd.concat(X_train).reset_index(drop=True)
    X_test = pd.concat(X_test).reset_index(drop=True)

    Data.dump(X_train, '../input/X_train_fe002.pkl')
    Data.dump(X_test, '../input/X_test_fe002.pkl')

    y_train = pd.Series([1 if i < 400 else 0 for i in range(1528)])
    Data.dump(y_train, '../input/y_train_fe002.pkl')
