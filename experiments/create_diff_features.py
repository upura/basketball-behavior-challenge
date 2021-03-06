import pandas as pd
from tqdm import tqdm

from ayniy.utils import Data


def extract_feat(sid: str, cat: str = 'train'):
    df = pd.read_csv(f'../input/{cat}/{sid}.csv')

    df['diff_usr_ball'] = ((df['usr_x'] - df['bal_x'])**2 + (df['usr_y'] - df['bal_y'])**2)
    df['diff_usr_uDF'] = ((df['usr_x'] - df['uDF_x'])**2 + (df['usr_y'] - df['uDF_y'])**2)
    df['diff_usr_scr'] = ((df['usr_x'] - df['scr_x'])**2 + (df['usr_y'] - df['scr_y'])**2)
    df['diff_scr_uDF'] = ((df['scr_x'] - df['uDF_x'])**2 + (df['scr_y'] - df['uDF_y'])**2)
    df['diff_scr_ball'] = ((df['scr_x'] - df['bal_x'])**2 + (df['scr_y'] - df['bal_y'])**2)
    df['diff_uDF_ball'] = ((df['uDF_x'] - df['bal_x'])**2 + (df['uDF_y'] - df['bal_y'])**2)

    use_cols = ['diff_usr_ball', 'diff_usr_uDF',
                'diff_usr_scr', 'diff_scr_uDF',
                'diff_scr_ball', 'diff_uDF_ball']

    df[use_cols].to_csv(f'../input/{cat}/{sid}_feat.csv', index=False)
    return (df['diff_usr_ball'].min(),
            df['diff_usr_uDF'].min(),
            df['diff_usr_scr'].min(),
            df['diff_scr_uDF'].min(),
            df['diff_scr_ball'].min(),
            df['diff_uDF_ball'].min(),
            len(df))


if __name__ == '__main__':

    X_train = []
    X_test = []

    for i in tqdm(range(1528)):
        X_train.append(extract_feat(sid=str(i).zfill(4), cat='train'))
    for i in tqdm(range(382)):
        X_test.append(extract_feat(sid=str(i).zfill(4), cat='test'))

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    col_names = ['min_diff_usr_ball',
                 'min_diff_usr_uDF',
                 'min_diff_usr_scr',
                 'min_diff_scr_uDF',
                 'min_diff_scr_ball',
                 'min_diff_uDF_ball',
                 'num_df']
    X_train.columns = col_names
    X_test.columns = col_names
    Data.dump(X_train, '../input/X_train.pkl')
    Data.dump(X_test, '../input/X_test.pkl')

    y_train = pd.Series([1 if i < 400 else 0 for i in range(1528)])
    Data.dump(y_train, '../input/y_train.pkl')
