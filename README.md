# DeepCC-local

This repo is based on Ergys Ristani's DeepCC \[[code](https://github.com/ergysr/DeepCC), [CVPR 2018 paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)\]. This tracker is based on *MATLAB*.

We added multiple functions for performance and utilities, including our *locality-aware* setting reported in our CVPR 2019 workshop paper (to be released). 

Besides, other dataset support are also added including MOT-16 and AI-City 2019.

# AI-City 2019 update

### Setup
For AI-City setup, please download the folder from [google drive](https://drive.google.com/drive/folders/1BklU8afXHoLu3xmmOcSFqniD3ZUJqqfJ?usp=sharing). Note that the official AI-City 2019 track-1 dataset also has to be downloaded. This folder only act as a incremental package. 

The folder we provide contains the re-ID features for demo usage. 

Before running, please check that the dataset position in `get_opts_aic.m` is changed as your setting.
```
opts.dataset_path    = '~/Data/AIC19';
```

After that, open up *MATLAB* at the code root directory, first run `get_opts_aic.m` to finish the setup. Then, type to run `add_gps.m` to add gps position to the detections. 


### Running Demo 
To run the demo, please open up *MATLAB* and run `val_aic_ensemble.m`. This should give you 79.7 IDF1 on the `train` set. 

For the `test` set, please run `test_aic_ensemble.m`. However, the test set result must be uploaded to the AI-City server for online test. To do that, please run `prepareMOTChallengeSubmission_aic.m`.

### Train your own re-ID model and run the tracker
If you want to train your own re-ID model, please check our other repo [open-reid-tracking](https://github.com/hou-yz/open-reid-tracking).

After training the re-ID model and computing the re-ID features for detection bounding boxes (pre-requisite of tracking), please run the `view_appear_score.m` file to get your own threshold/norm parameters. 
NOthe that the experiment directory in `view_appear_score.m` must be changed accordingly before running. 
```
opts.net.experiment_root = 'experiments/zju_lr001_colorjitter_256_gt_val';
```

After that, you can replace the old parameters. Remember to change the new feature saving directory in `val_aic_ensemble.m` or `test_aic_ensemble.m`, and you should be good to go. 
