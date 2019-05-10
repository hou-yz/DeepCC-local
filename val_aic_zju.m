clear
clc

%% Options
opts = get_opts_aic();
opts.experiment_name = 'aic_zju';
% opts.detections = 'yolo3';
% basis setting for DeepCC
opts.tracklets.window_width = 10;
opts.trajectories.window_width = 30;
opts.identities.window_width = [500,4800];
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.feature_dir = 'det_features_zju_lr001_test_ssd';

%% lr001 
% 4.5/4.9/5.3
% s02: 81.4/72.6; s134: 81.6/80.9
% 3.9/4.1/5.3
% s02: 59.5/58.5; s134: 82.9/82.2

% fix acute cam:     72->74
% speed change < 80: 72->74
% fix accute + < 80: 72->77

create_experiment_dir(opts);
%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker
% opts.visualize = true;
opts.sequence = 3;

%% GRID SEARCH
thresholds = 1;%5.5:-0.2:3.5;
l2_scts = zeros(length(thresholds),3);
removed_scts = zeros(length(thresholds),3);
l3_scts = zeros(length(thresholds),3);
l3_mcts = zeros(length(thresholds),3);
for i = 1:length(thresholds)
thres = thresholds(i);

opts.tracklets.threshold    = 4.5;
opts.trajectories.threshold = 4.5;
opts.identities.threshold   = 4.9;
opts.tracklets.diff_p    = 1.82;
opts.trajectories.diff_p = 1.82;
opts.identities.diff_p   = 1.82;
opts.tracklets.diff_n    = 1.82;
opts.trajectories.diff_n = 1.82;
opts.identities.diff_n   = 1.82;

% alpha
% opts.tracklets.alpha    = 1;
% opts.trajectories.alpha = 1;
% opts.identities.alpha   = 1;

%% Tracklets
% opts.tracklets.spatial_groups = 0;
% opts.optimization = 'KL';
% compute_L1_tracklets_aic(opts);
% 
% %% Single-camera trajectories
% opts.trajectories.appearance_groups = 0;
% compute_L2_trajectories_aic(opts);
% opts.eval_dir = 'L2-trajectories';
% [~, metsSCT, ~] = evaluate(opts);
% l2_scts(i,:) = metsSCT(1:3);
% 
% %% remove waiting cars
% removeOverlapping(opts);
% opts.eval_dir = 'L2-removeOvelapping';
% [~, metsSCT, ~] = evaluate(opts);
% removed_scts(i,:) = metsSCT(1:3);

%% Multi-camera identities
opts.identities.consecutive_icam_matrix = ones(40);
opts.identities.reintro_time_matrix = ones(1,40)*inf;
opts.identities.appearance_groups = 0;
compute_L3_identities_aic(opts);
opts.eval_dir = 'L3-identities';
[~, metsSCT, metMCT] = evaluate(opts);
l3_scts(i,:) = metsSCT(1:3);
l3_mcts(i,:) = metMCT;
end