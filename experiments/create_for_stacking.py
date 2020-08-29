import pandas as pd
from scipy.stats import rankdata

from ayniy.utils import Data


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


run_ids = [
    'weight002',
    'weight004'
]

y_train = Data.load('../input/y_train_fe000.pkl')
data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]

train = [pd.Series(d[0]) for d in data]
test = [pd.Series(d[1]) for d in data]

X_train = pd.concat(train, axis=1)
X_test = pd.concat(test, axis=1)
X_train.columns = run_ids
X_test.columns = run_ids

fe_name = 'fe006'
Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
