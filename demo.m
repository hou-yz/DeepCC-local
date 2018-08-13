%% Options
opts = get_opts();
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
opts.identities.appearance_groups = 0;
compute_L3_identities(opts);
opts.eval_dir = 'L3-identities';
evaluate(opts);

