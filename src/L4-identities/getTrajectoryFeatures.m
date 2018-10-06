function trajectories = getTrajectoryFeatures(opts, trajectories)

% Gather bounding boxes for this trajectory
detections = [];
csvfile = sprintf('%s/%s/L4-identities/temp_images/L2images.csv',opts.experiment_root, opts.experiment_name);


if opts.identities.extract_images
    temp_dir = sprintf('%s/%s/L4-identities/temp_images_%s',opts.experiment_root, opts.experiment_name, opts.sequence_names{opts.sequence});
    status = rmdir(temp_dir,'s');
    mkdir(temp_dir);
    fid = fopen(csvfile,'w');
end

for i = 1:length(trajectories)
    
    
    % For every trajectory, extract one image every k frames
    inds = round(linspace(1,size(trajectories(i).trajectories(1).data,1),10));
    
    if opts.identities.extract_images
        fprintf('Extracting imgages for trajectory %d/%d\n', i, length(trajectories));
        mkdir(sprintf('%s/%s/L4-identities/temp_images_%s/%05d/',opts.experiment_root, opts.experiment_name, opts.sequence_names{opts.sequence},   i));
    end
    
    for k=1:length(inds)
        bb       = trajectories(i).trajectories(1).data(inds(k),3:6);
        frame    = trajectories(i).trajectories(1).data(inds(k),1);
        camera   = trajectories(i).trajectories(1).camera;
        if opts.identities.extract_images
            img      = opts.reader.getFrame(camera, frame);
            snapshot = get_bb(img,bb);
            snapshot = imresize(snapshot,[opts.net.input_height, opts.net.input_width]);
            % trajectories(i).trajectories(1).snapshot{k} = snapshot;
            filename = sprintf('%s/%s/L4-identities/temp_images_%s/%05d/img_%d_%d.jpg',opts.experiment_root, opts.experiment_name, opts.sequence_names{opts.sequence} , i,i,k);
            imwrite(snapshot, filename);
            fprintf(fid,'%05d,%s\n',k,filename);
        end
        detections = [detections; camera, frame, bb, i, k];
    end
end
if opts.identities.extract_images
    fclose(fid);
end

% Compute features
% features = embed_detections(opts, detections);
net = opts.net;
cur_dir = pwd;
cd src/triplet-reid
featuresfile = sprintf('%s/%s/L4-identities/L2features_%s.h5',opts.experiment_root, opts.experiment_name, opts.sequence_names{opts.sequence});

command = strcat('python embed.py' , ...
    sprintf(' --experiment_root %s', net.experiment_root), ...
    sprintf(' --image_root %s', fullfile(cur_dir)), ...
    sprintf(' --filename L2features_%s.h5', opts.sequence_names{opts.sequence}), ...
    sprintf(' --dataset ../../%s', csvfile));
fprintf(command);
system(command);
cd(cur_dir);
movefile(sprintf('src/triplet-reid/%s/L2features_%s.h5',opts.net.experiment_root, opts.sequence_names{opts.sequence}),featuresfile);

features = h5read(featuresfile, '/emb');
features = features';

% Assign features to trajectories
ids = unique(detections(:,7));
for i = 1:length(ids)
    id = ids(i);
    trajectory_features = features(detections(:,7)==id,:);
    % Trajectory feature is the average of all features (if more than one image)
    trajectories(id).trajectories(1).feature = mean(trajectory_features,1);
end

