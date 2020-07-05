import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import QuantileTransformer
import torch
from torch.utils.data import DataLoader

from src.datasets import BasketDataset
from src.models import BasketNN
from src.utils import seed_everything
from src.runner import CustomRunner

from ayniy.utils import Data


if __name__ == '__main__':

    run_name = 'nn001'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_train = Data.load('../input/X_train.pkl')
    y_train = Data.load('../input/y_train.pkl')
    X_test = Data.load('../input/X_test.pkl')

    # rankgauss transform
    # https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
    prep = QuantileTransformer(output_distribution="normal")
    X_train = pd.DataFrame(prep.fit_transform(X_train))
    X_test = pd.DataFrame(prep.transform(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    cv_scores = []

    test_dataset = BasketDataset(X=X_test, y=None)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=2000)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):

        X_tr = X_train.loc[tr_idx, :]
        X_val = X_train.loc[va_idx, :]
        y_tr = y_train[tr_idx]
        y_val = y_train[va_idx]

        train_dataset = BasketDataset(X=X_tr, y=y_tr)
        valid_dataset = BasketDataset(X=X_val, y=y_val)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2000)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=2000)

        loaders = {'train': train_loader, 'valid': valid_loader}
        runner = CustomRunner(device=device)

        model = BasketNN(
            in_channels=10,
            n_cont_features=X_tr.shape[1],
            hidden_channels=64,
            kernel_sizes=[3, 5, 7, 15, 21, 51],
            out_dim=1,
        )
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        logdir = f'../output/logdir_{run_name}/fold{fold_id}'
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=20,
            verbose=True,
        )

        pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                       runner.predict_loader(
                                           loader=valid_loader,
                                           resume=f'{logdir}/checkpoints/best.pth',
                                           model=model,),)))

        oof_preds[va_idx] = pred
        y_pred_oof = (pred > 0.5).astype(int)
        score = accuracy_score(y_val, y_pred_oof)
        cv_scores.append(score)
        print('score', score)

        pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                       runner.predict_loader(
                                       loader=test_loader,
                                       resume=f'{logdir}/checkpoints/best.pth',
                                       model=model,),)))
        test_preds += pred / 5

    # save results
    print(cv_scores)
    pd.Series(oof_preds).to_csv(f'../output/pred/oof_{run_name}.csv', index=False, header=None)
    pd.Series(test_preds).to_csv(f'../output/pred/pred_{run_name}.csv', index=False, header=None)

    y_pred_test = (test_preds > 0.5).astype(int)
    pd.Series(y_pred_test).to_csv(f'../output/submissions/test_prediction_{run_name}.csv', index=False, header=None)
