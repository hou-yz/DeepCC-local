%% Options
opts = get_opts();
opts.experiment_name = 'paper_result';
create_experiment_dir(opts);

%% Setup Gurobi
if ~exist('setup_done','var')
    setup;
    setup_done = true;
end

%% Run Tracker

opts.eval_dir = 'DeepCC';
evaluate(opts);

