clear
clc

%% Options
opts = get_opts_aic();
opts.experiment_name = 'aic_zju_ensemble';
% basis setting for DeepCC
opts.tracklets.window_width = 10;
opts.trajectories.window_width = 30;
opts.identities.window_width = [500,4800];
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.feature_dir = 'det_features_zju_lr001_ensemble_trainval_ssd';

%% lr001 ensemble
% 0.65/0.71/0.73
% s02: 83.9/80.0; s134: 
% fix accute + < 80: 75->80

create_experiment_dir(opts);
%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker
opts.optimization = 'KL';
% opts.visualize = true;
% 'train_134', 'test_6', 'test_2', 'test_5', 'test_26','test_25','train_34','train_1'
opts.sequence = 1;

% %% GRID SEARCH
% thresholds = 1%0.8:-0.03:0.5;
% l2_scts = zeros(length(thresholds),3);
% removed_scts = zeros(length(thresholds),3);
% l3_scts = zeros(length(thresholds),3);
% l3_mcts = zeros(length(thresholds),3);
% for i = 1:length(thresholds)
% thres = thresholds(i);

opts.tracklets.threshold    = 0.65;
opts.trajectories.threshold = 0.65;
opts.identities.threshold   = 0.71;
opts.tracklets.diff_p    = 0.26;
opts.trajectories.diff_p = 0.26;
opts.identities.diff_p   = 0.26;
opts.tracklets.diff_n    = 0.26;
opts.trajectories.diff_n = 0.26;
opts.identities.diff_n   = 0.26;

%% Tracklets
opts.tracklets.spatial_groups = 0;
compute_L1_tracklets_aic(opts);

%% Single-camera trajectories
opts.trajectories.appearance_groups = 1;
compute_L2_trajectories_aic(opts);
opts.eval_dir = 'L2-trajectories';
[~, metsSCT, ~] = evaluate(opts);
% l2_scts(i,:) = metsSCT(1:3);

%% remove waiting cars
removeOverlapping(opts);
opts.eval_dir = 'L2-removeOverlapping';
[~, metsSCT, ~] = evaluate(opts);
% removed_scts(i,:) = metsSCT(1:3);

%% Multi-camera identities
opts.identities.consecutive_icam_matrix = ones(40);
opts.identities.reintro_time_matrix = ones(1,40)*inf;
opts.identities.appearance_groups = 0;
compute_L3_identities_aic(opts);
opts.eval_dir = 'L3-identities';
[~, metsSCT, metsMCT] = evaluate(opts);
% l3_scts(i,:) = metsSCT(1:3);
% l3_mcts(i,:) = metsMCT;
% end