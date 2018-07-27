%% Run Tracker

opts = get_opts();

for iSequence = 3:4
	opts.sequence = iSequence;

	% Tracklets
	compute_L1_tracklets(opts);

	% Single-camera trajectories
	opts.trajectories.appearance_groups = 1;
	compute_L2_trajectories(opts);

	% Multi-camera identities
	opts.identities.appearance_groups = 0;
	compute_L3_identities(opts);

end

%% Prepare Submission file duke.zip