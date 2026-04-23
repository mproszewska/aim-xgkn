# XGKN

### Datasets
Datasets are downloaded using `torch geometric`. 
The data splits are the same as in [here](https://github.com/diningphil/gnn-comparison).

### Training
Run
```
python train.py --dataset MUTAG
```
For datasets with predefined splits use `--split`, for others use `--seed` for random splits.

### Evaluation
Check out `../AIM` dir.
