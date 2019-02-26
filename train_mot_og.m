clear
clc

%% Options
opts = get_opts_mot();
opts.experiment_name = 'mot_og';
% basis setting for DeepCC
opts.tracklets.window_width = 20;
opts.trajectories.window_width = 80;
opts.trajectories.overlap = 40;
% correlation threshold setting according to `view_distance_distribution(opts)`
opts.tracklets.threshold    = 1;
opts.trajectories.threshold = 1;

opts.tracklets.diff_p    = 0.64;
opts.trajectories.diff_p = 0.64;
opts.tracklets.diff_n    = 0.64;
opts.trajectories.diff_n = 0.64;

% alpha
opts.tracklets.alpha    = 0;
opts.trajectories.alpha = 1;

create_experiment_dir(opts);

%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker

% opts.visualize = true;

% Tracklets
compute_L1_tracklets_mot(opts);

% Single-camera trajectories
opts.trajectories.appearance_groups = 1;
compute_L2_trajectories_mot(opts);
evaluate(opts,1);