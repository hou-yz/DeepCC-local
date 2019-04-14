function opts = get_opts_aic()

addpath(genpath('src'))

opts = [];
opts.dataset = 2;%0 for duke, 1 for mot, 2 for aic
opts.feature_dir     = [];
opts.dataset_path    = 'D:/Data/AIC19';
opts.gurobi_path     = 'C:/Utils/gurobi801/linux64/matlab';
opts.experiment_root = 'experiments';
opts.experiment_name = 'aic_demo';
[opts.projection,opts.fisheye_mapping] = get_projection_param(opts);

opts.reader = MyVideoReader_aic(opts.dataset_path);

% General settings
opts.eval_dir = 'L3-identities';
opts.visualize = false;
opts.image_width = 1920;
opts.image_height = 1080;
opts.current_camera = -1;
opts.minimum_trajectory_length = 5;
opts.optimization = 'KL'; 
opts.sequence = 1;
opts.sequence_names = {'trainval', 'trainval_mini', 'test_easy', 'test_hard', 'trainval_nano','test_all','train','val'};
opts.seqs = {[1,3,4],[],[],[],[],[2,5],[3,4],1};
opts.scenarios = 1:5;
opts.cams_in_scene = {1:5,6:9,10:15,16:40,[10,16:29,33:36]};
opts.sub_dir = {'train', 'train', '', '', '', 'test', '', 'train'};
opts.trainval_scene_by_icam = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4];
opts.render_threshold = -10;
opts.load_tracklets = 1;
opts.load_trajectories = 1;
% opts.appear_model_name = 'MOT/og512/model_1param_L2_15.mat';
opts.appear_model_name = '1fps_train_IDE_40/model_param_L2_75.mat';
opts.motion_model_name = '1fps_train_IDE_40/model_param_L2_motion_150.mat';
opts.soft = 0.1;
opts.fft = false;
opts.fps = 10;
opts.frame_offset = get_frame_offset(opts);
opts.world_center = {[42.525678, -90.723601],[42.491916, -90.723723],[42.498780, -90.686393],[42.498780, -90.686393],[42.498780, -90.686393]};
opts.world_scale = 6371000;


% Tracklets
tracklets = [];
tracklets.spatial_groups = 1;
tracklets.window_width = 10;
tracklets.min_length = 2;
tracklets.alpha = 1;
tracklets.beta = 0.02;
tracklets.cluster_coeff = 0.75;
tracklets.nearest_neighbors = 8;
tracklets.speed_limit = 500;
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
trajectories.beta = 0.001;
trajectories.window_width = 30;
trajectories.speed_limit = 500;
trajectories.indifference_time = 100;
trajectories.threshold = 8;
trajectories.diff_p = 0;
trajectories.diff_n = 0;
trajectories.step = false;
trajectories.og_appear_score = true;
trajectories.og_motion_score = true;
trajectories.use_indiff = true;
trajectories.smoothness_interval_length = 4;
trajectories.weightSmoothness = 0;
trajectories.weightVelocityChange = 0;
trajectories.weightTimeInterval = 0;
trajectories.weightIOU = 0;

% Identities
identities = [];
identities.window_width = 200;
identities.appearance_groups = 0; % determined automatically when zero
identities.alpha = 2;
identities.beta = 0.01;
identities.overlap = 5;
identities.speed_limit = 2000;
identities.indifference_time = 150;
identities.threshold = 8;
identities.diff_p = 0;
identities.diff_n = 0;
identities.optimal_filter = true;
identities.step = false;
identities.extract_images = true;
identities.og_appear_score = true;
identities.og_motion_score = true;

opts.tracklets = tracklets;
opts.trajectories = trajectories;
opts.identities = identities;
end

