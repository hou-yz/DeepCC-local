%% Options
opts = get_opts();
opts.experiment_name = '1fps_L2_motion';
opts.feature_dir = 'det_features_fc256_train_1fps_trainBN_trainval';
% basis setting for DeepCC
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
opts.trajectories.overlap = 75;
opts.identities.window_width = 12000;
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.tracklets.threshold    = 18.45;
opts.trajectories.threshold = 18.45;%0.468; 
opts.identities.threshold   = 0;%0.486;

opts.tracklets.diff_p    = 11;
opts.trajectories.diff_p = 11;
opts.identities.diff_p   = 0.1;
opts.tracklets.diff_n    = 11;
opts.trajectories.diff_n = 11;
opts.identities.diff_n   = 0.1;

% alpha
opts.tracklets.alpha    = 0;
opts.trajectories.alpha = 1;
opts.identities.alpha   = 0;

create_experiment_dir(opts);

%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker

% opts.visualize = true;
opts.sequence = 8; % val

opts.appear_model_name = '1fps_train_IDE_40/model_param_L2_75.mat';
% Tracklets
opts.optimization = 'KL';
%compute_L1_tracklets(opts);

% Single-camera trajectories
%opts.trajectories.use_indiff = false;
% opts.trajectories.og_appear_score = false;
% opts.trajectories.og_motion_score = false;
opts.trajectories.appearance_groups = 1;
compute_L2_trajectories(opts);
opts.eval_dir = 'L2-trajectories';
evaluate(opts);

%opts.appear_model_name = '1fps_train_IDE_40/model_param_L3_3000.mat';
%opts.soft = 1;

% % Multi-camera identities
% %opts.optimization = 'BIPCC';
% opts.identities.optimal_filter = false;
% opts.identities.og_appear_score = false;
% opts.identities.consecutive_icam_matrix = ones(8);
% opts.identities.reintro_time_matrix = ones(1,8)*inf;
% 
% opts.identities.appearance_groups = 0;
% compute_L3_identities(opts);
% opts.eval_dir = 'L3-identities';
% evaluate(opts);
