clear
clc

%% Options
opts = get_opts_aic();
opts.experiment_name = 'aic_og';
opts.feature_dir = 'det_features_ide_basis_train_5fps_val';
% basis setting for DeepCC
opts.tracklets.window_width = 10;
opts.trajectories.window_width = 30;
opts.trajectories.overlap = 15;
opts.identities.window_width = 200;
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.tracklets.threshold    = 11.91;
opts.trajectories.threshold = 11.91;
opts.identities.threshold   = 11.91;

opts.tracklets.diff_p    = 7.63;
opts.trajectories.diff_p = 7.63;
opts.identities.diff_p   = 7.63;
opts.tracklets.diff_n    = 7.63;
opts.trajectories.diff_n = 7.63;
opts.identities.diff_n   = 7.63;

% alpha
opts.tracklets.alpha    = 0;
opts.trajectories.alpha = 0;
opts.identities.alpha   = 0;

create_experiment_dir(opts);
%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker
% opts.visualize = true;
opts.sequence = 8;

% Tracklets
opts.tracklets.spatial_groups = 1;
opts.optimization = 'KL';
% compute_L1_tracklets_aic(opts);

% Single-camera trajectories
opts.optimization = 'KL';
%opts.trajectories.use_indiff = false;
opts.experiment_name = 'aic_og';
opts.trajectories.appearance_groups = 0;    
% compute_L2_trajectories_aic(opts);
% opts.eval_dir = 'L2-trajectories';
% evaluate(opts);

% Multi-camera identities
% opts.optimization = 'BIPCC';
opts.identities.optimal_filter = true;
opts.identities.consecutive_icam_matrix = ones(40);
opts.identities.reintro_time_matrix = ones(1,40)*inf;
opts.identities.appearance_groups = 0;
compute_L3_identities_aic(opts);
opts.eval_dir = 'L3-identities';
evaluate(opts);
