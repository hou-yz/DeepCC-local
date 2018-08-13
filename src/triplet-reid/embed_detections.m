function features = embed_detections(opts, detections)
% Computes feature embeddings for given detections
% Detections are in format [cam, frame, left, top, width, height]
net = opts.net;

% Detection images read in python and embedded 
cur_dir = pwd;
cd src/triplet-reid

% Temporary file to store the bounding box coordinates
features_filename = fullfile(net.experiment_root, 'temp_features.h5');
detections_filename = fullfile(net.experiment_root, 'temp_detections.mat');
save(detections_filename, 'detections');

command = strcat('python embed_detections.py' , ...
    sprintf(' --experiment_root %s', net.experiment_root), ...
    sprintf(' --dataset_path %s', opts.dataset_path), ...
    sprintf(' --detections_path %s', detections_filename), ...
    sprintf(' --filename %s', features_filename));
system(command);

% Load features / delete temp files
features = h5read(features_filename, '/emb');
delete(detections_filename);
delete(features_filename);
cd(cur_dir)