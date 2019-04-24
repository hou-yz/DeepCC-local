clear
clc

%% Options
opts = get_opts_aic();
opts.experiment_name = 'aic_tc_ssd';
opts.sequence = 8;
opts.eval_dir = 'L2-trajectories';
evaluate(opts);
