from src.datasets import BasketDataset

from ayniy.utils import Data


X_train = Data.load('../input/X_train.pkl')
y_train = Data.load('../input/y_train.pkl')
X_test = Data.load('../input/X_test.pkl')

print(max(X_train['num_df']))
print(max(X_test['num_df']))

test_dataset = BasketDataset(X=X_test, y=None)

print(test_dataset)
print(test_dataset[0])
