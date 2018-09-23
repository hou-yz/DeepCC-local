%% Options
opts = get_opts();
opts.experiment_name = 'fc256_30fps_separate_icam_fake_50x';
opts.feature_dir = 'det_features_fc256_30fps_separate_icam_fake_trainval_mini';
% basis setting for DeepCC
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
opts.trajectories.overlap = 75;
opts.identities.window_width = 6000;
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.tracklets.threshold = [19.77,18.89,15.00,11.78,14.62,18.92,15.26,17.61];%FPs+50*FNs
opts.trajectories.threshold = [19.77,18.89,15.00,11.78,14.62,18.92,15.26,17.61];%FPs+50*FNs

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

