function opts = get_opts_aic()

addpath(genpath('src'))

opts = [];
opts.feature_dir     = 'D:/Data/AIC19/det_feat/og512/';
opts.dataset_path    = 'D:/Data/AIC19/';
opts.gurobi_path     = 'C:/Utils/gurobi801/win64/matlab';
opts.experiment_root = 'experiments';
opts.experiment_name = 'demo';


% General settings
opts.eval_dir = 'L2-trajectories';
opts.visualize = false;
opts.image_width = 1920;
opts.image_height = 1080;
opts.current_camera = -1;
opts.minimum_trajectory_length = 50;
opts.optimization = 'KL'; 
opts.use_groupping = 1;
opts.sequence = 7;
opts.sequence_names = {'trainval', 'trainval_mini', 'test_easy', 'test_hard', 'trainval_nano','test_all','train','val'};
opts.seqs = [2,4,5,9,10,11,13];
opts.render_threshold = -10;
opts.load_tracklets = 1;
opts.load_trajectories = 1;
% opts.appear_model_name = 'MOT/og512/model_param_L2_15.mat';
opts.soft = 0.1;

% Tracklets
tracklets = [];
tracklets.window_width = 50;
tracklets.min_length = 5;
tracklets.alpha = 1;
tracklets.beta = 0.02;
tracklets.cluster_coeff = 0.75;
tracklets.nearest_neighbors = 8;
tracklets.speed_limit = 20;
tracklets.threshold = 8;
tracklets.diff_p = 0;
tracklets.diff_n = 0;
tracklets.step = false;
tracklets.og_appear_score = true;
tracklets.og_motion_score = true;


% Trajectories
trajectories = [];
trajectories.appearance_groups = 0; % determined automatically when zero
trajectories.alpha = 1;
trajectories.beta = 0.01;
trajectories.window_width = 300;
trajectories.speed_limit = 30;
trajectories.indifference_time = 100;
trajectories.threshold = 8;
trajectories.diff_p = 0;
trajectories.diff_n = 0;
trajectories.step = false;
trajectories.og_appear_score = true;
trajectories.og_motion_score = true;
trajectories.use_indiff = true;

opts.tracklets = tracklets;
opts.trajectories = trajectories;
end

