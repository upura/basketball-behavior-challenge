cd experiments

python create_tsfresh_features.py   # Extract tsfresh features
python change_colnames.py           # Rename some column names for avoid error in LGBM
python create_diff_features.py      # Extract features for 1DCNN
python train_nn.py                  # Run 1DCNN
python concat_features.py           # Concat features of tsfresh and 1DCNN

python runner.py --run configs/run004.yml   # Run LGBM for tsfresh
python runner.py --run configs/run006.yml   # Run LGBM for tsfresh + 1DCNN
python create_for_stacking.py               # Concat 2 predictions for stacking
python runner.py --run configs/run008.yml   # Run Ridge for stacking
python weighted_averaging.py                # Weighted averaging and tune threshold

cd ../
