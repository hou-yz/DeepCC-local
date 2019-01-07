clc
clear

opts=get_opts();
opts.experiment_name = '1fps_og';
opts.feature_dir = 'det_features_fc256_train_1fps_trainBN_trainval';
% basis setting for DeepCC
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
opts.trajectories.overlap = 75;
opts.identities.window_width = 6000;


opts.sequence = 7;

filename = sprintf('%s/%s/L3-identities/L2trajectories.mat',opts.experiment_root, opts.experiment_name);
load(filename);