%% Options
opts = get_opts();
opts.experiment_name = 'fc256_30fps_separate_icam';
opts.feature_dir = 'det_features_fc256_30fps_separate_icam_trainval_mini';
% basis setting for DeepCC
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
opts.trajectories.overlap = 75;
opts.identities.window_width = 6000;
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.tracklets.threshold = [15.14,14.49,13.52,11.49,12.72,13.97,13.61,13.62];
opts.trajectories.threshold = [15.14,14.49,13.52,11.49,12.72,13.97,13.61,13.62];

create_experiment_dir(opts);

%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker

% opts.visualize = true;
opts.sequence = 2; % trainval-mini

% Tracklets
opts.optimization = 'KL';
compute_L1_tracklets(opts);

% Single-camera trajectories
opts.optimization = 'BIPCC';
opts.trajectories.appearance_groups = 1;
compute_L2_trajectories(opts);
opts.eval_dir = 'L2-trajectories';
evaluate(opts);

% Multi-camera identities
%opts.identities.appearance_groups = 0;
%compute_L3_identities(opts);
%opts.eval_dir = 'L3-identities';
%evaluate(opts);

