import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, accuracy_score

from ayniy.utils import Data


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


def f(x):
    pred = 0
    for i, d in enumerate(data):
        if i < len(x):
            pred += d[0] * x[i]
        else:
            pred += d[0] * (1 - sum(x))
    score = roc_auc_score(y_train, pred) * -1
    Data.dump(pred, f'../output/pred/{run_name}-train.pkl')
    return score


def g(x):
    score = accuracy_score(y_train, oof > x) * -1
    return score


def make_predictions(data: list, weights: list, is_oof=True):
    pred = 0
    if is_oof:
        for i, d in enumerate(data):
            if i < len(weights):
                pred += d[0] * weights[i]
            else:
                pred += d[0] * (1 - sum(weights))
    else:
        for i, d in enumerate(data):
            if i < len(weights):
                pred += d[1] * weights[i]
            else:
                pred += d[1] * (1 - sum(weights))
        Data.dump(pred, f'../output/pred/{run_name}-test.pkl')
    return pred


def make_submission(pred, run_name: str, th: float):
    pred = (pred > th).astype(int)
    pd.Series(pred).to_csv(f'../output/submissions/test_prediction_{run_name}.csv', index=False)


run_ids = [
    # 'run001',
    'run003',
]
run_name = 'weight001'

y_train = Data.load('../input/y_train_fe000.pkl')
data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]

for d in data:
    print(roc_auc_score(y_train, d[0]))

init_state = [round(1 / len(data), 3) for _ in range(len(data) - 1)]
result = minimize(f, init_state, method='Nelder-Mead')
print('optimized CV: ', result['fun'])
print('w: ', result['x'])

oof = make_predictions(data, result['x'], is_oof=True)
pred = make_predictions(data, result['x'], is_oof=False)

binary_result = minimize(g, 0.5, method='Nelder-Mead')
print(binary_result)
make_submission(pred, run_name, th=binary_result['x'][0])
