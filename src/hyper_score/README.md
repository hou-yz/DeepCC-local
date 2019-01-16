# metric learning for temporal locality

Metric Net Architecture:
- Input: `256-dim` absolute difference between a pair of IDE features.
- Output: `2-dim` softmax classification result.
- Hidden layers: 3x `128-dim` hidden layers.

Dataset & input: 
- Dataset: 40-frame-long tracklet appearance feature (IDE feature). Use the first 40 minute as `train` and the rest 11 minute as `val`.
- Sampling: select IDE feature pair within
  - Temporal window.
  - Saptial window (camera).

Use cross-entropy loss for training. Train 30 epochs with `lr=1e-3`, and another 10 epochs with `lr=1e-4`. Use SGD optimizer with `momentum=0.5`.

L2-norm (L2-norm) distance metric is used in tracklets forming.

Within-Camera Metric Learning (M2) with settings:
- 75-frame-long temporal window; same camera.
- `weight_decay=5e-4`.

Cross-Camera Metric Learning (M3) with settings:
- 12000-frame-long temporal window; all 8 cameras.
- `weight_decay=5e-2`.


The `IDF-1` results on `val` are as follows. 

|                       | SCT (%) | MCT (%) |
| ---                   | :---: | :---: |
| (L2-norm)/M2/M2       | 87.74 | 83.61 |
| (L2-norm)/M2/M3       | 87.71 | 84.01 |