function create_experiment_dir(opts)
warning off;
mkdir([opts.experiment_root, filesep, opts.experiment_name]);
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, 'L0-features']);
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, 'L1-tracklets']);
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, 'L2-trajectories']);
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, 'L3-identities']);
