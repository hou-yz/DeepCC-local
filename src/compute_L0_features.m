function compute_L0_features(opts)
% Computes features for the input poses

for iCam = 1:opts.num_cam
    
    % Load poses
    load(fullfile(opts.dataset_path, 'detections','openpose', sprintf('camera%d.mat',iCam)));
    poses = detections;
    
    % Convert poses to detections
    detections = zeros(size(poses,1),6);
    for k = 1:size(poses,1)
        pose = poses(k,3:end);
        bb = pose2bb(pose, opts.render_threshold);
        [newbb, newpose] = scale_bb(bb,pose,1.25);
        detections(k,:) = [iCam, poses(k,2), newbb];
    end
    
    % Compute feature embeddings
    features = embed_detections(opts,detections);
    
    % Save features
    h5write(sprintf('%s/%s/L0-features/features%d.h5',opts.experiment_root,opts.experiment_name,iCam),'/emb', features);
end

