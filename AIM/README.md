# Training models

### GCN, GAT and GIN
Use `train_gnns.py`, e.g.
```bash
python train_gnns.py --model_type GCN --dataset MUTAG
```

### ProtGNN

Use [ProtGNN](https://github.com/zaixizhang/ProtGNN/tree/main) repository to train ProtGNN. Ensure that datasets are preprocessed the same way for AIM eval.

### GKNN

Use code from Supplementary Material on [openreview](https://openreview.net/forum?id=5fbUEUTZEn7) (or [Graph-Kernel-Convolution](https://github.com/gdl-unive/Graph-Kernel-Convolution)).

### KerGNN

Use [KerGNN](https://github.com/asFeng/kergnns) repository to train KerGNN models.

### XGKN

Check out `../XGKN` dir.

# AIM evaluation

Each model type has a separate evaluation script to avoid library version conflicts, though the scripts are largely similar.

### Most relevant args
Use `--node_mask_fn` or `--edge_mask_fn` to define thresholding technique e.g. `--node_mask_fn percentile:0.8`, `--node_mask_fn elbow:0`. 
Use `--metrics` to choose metrics to run, e.g. `--metrics A A1 A1` will only compute predictive accuracy, and metrics A1-A2. 
All metrics are enabled by default.

### XGKN
```
python eval_xgkn.py --dataset MUTAG --model_path ${CHECKPOINT} --explainer SHAP --node_mask_fn: percentile_softmax:0.8
```

### GIN and GAT
```
python eval_explainer.py --dataset MUTAG --model_path ${CHECKPOINT} --explainer IntegratedGradients --node_mask_fn: percentile_softmax:0.8
```

### ProtGNN
```
python eval_protgnn.py --dataset MUTAG --model_path ${CHECKPOINT} --explainer ShapleyValueSampling --node_mask_fn: percentile_softmax:0.8
```
Use an absolute path for the checkpoint here.

### KerGNN
```
python eval_kergnn.py --dataset MUTAG --model_path ${CHECKPOINT} --explainer SHAP --node_mask_fn: percentile_softmax:0.8
```

### GKNN
```
python eval_gknn.py --dataset MUTAG --model_path ${CHECKPOINT} --explainer SHAP --node_mask_fn: percentile_softmax:0.8
```
