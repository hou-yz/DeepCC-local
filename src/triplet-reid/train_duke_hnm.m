opts = get_opts();

% Run few iterations to obtain a preliminary embedding
tmp = opts;
tmp.net.experiment_root = 'experiments/demo_hnm_tmp';
tmp.net.train_iterations = 5000;
train_duke(tmp);
embed(tmp);

% Train with hard negative mining
opts.net.experiment_root = 'experiments/demo_hnm';
opts.net.hard_pool_size = 50;
opts.net.train_embeddings = sprintf('%s/src/triplet-reid/%s/duke_train_embeddings.h5',pwd,tmp.net.experiment_root);  
train_duke(opts);
embed(opts);
evaluation_res_duke_fast(opts);