function compute_L1_tracklets(opts)
% Computes tracklets for all cameras
if opts.tracklets.og_appear_score
    appear_model_param = [];
else
    appear_model_param = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
end
if opts.tracklets.og_motion_score
    motion_model_param = [];
else
    motion_model_param = load(fullfile('src','hyper_score/logs',opts.motion_model_name));
end
for iCam = 1:8
    
    opts.current_camera = iCam;
    
    sequence_window   = opts.sequence_intervals{opts.sequence};
    start_frame       = global2local(opts.start_frames(opts.current_camera), sequence_window(1));
    end_frame         = global2local(opts.start_frames(opts.current_camera), sequence_window(end));
    
    % Load OpenPose detections for current camera
    detections = load(fullfile(opts.dataset_path, 'detections','OpenPose', sprintf('camera%d.mat',iCam)));
    detections = detections.detections;
    
    % Load features for all detections
    if isempty(opts.feature_dir)
        features   = h5read(sprintf('%s/L0-features/features%d.h5',opts.dataset_path,iCam),'/emb');
    else
        features   = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb');
    end
    in_time_range_ids = detections(:,2)>=start_frame & detections(:,2)<=end_frame;
    all_dets   = detections(in_time_range_ids,:);
    if size(features,2) == size(detections,1)
        features = features(3:end,in_time_range_ids);
    else
        features = features(3:end,features(2,:)>=start_frame & features(2,:)<=end_frame);
    end
    appearance = cell(size(all_dets,1),1);
    frames     = cell(size(all_dets,1),1);
    for k = 1:length(frames)
        appearance{k} = double(features(:,k)');
        frames{k} = detections(k,2);
    end
    
    % Compute tracklets for every 1-second interval
    tracklets = struct([]);
    
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        window_inds          = find(ismember(all_dets(:,2),window_frames));
        detections_in_window = all_dets(window_inds,:);
        detections_conf      = sum(detections_in_window(:,5:3:end),2);
        num_visible          = sum(detections_in_window(:,5:3:end)> opts.render_threshold, 2);
        
        % Use only valid detections
        [valid, detections_in_window] = getValidDetections(detections_in_window, detections_conf, num_visible, opts, iCam);
        detections_in_window          = detections_in_window(valid,:);
        detections_in_window(:,7:end) = [];
        detections_in_window(:,[1 2]) = detections_in_window(:,[2 1]);
        filteredDetections = detections_in_window;
        filteredFeatures = [];
        filteredFeatures.appearance = appearance(window_inds(valid));
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        tracklets = createTracklets(opts, filteredDetections, filteredFeatures, window_start_frame, window_end_frame, tracklets,appear_model_param,motion_model_param);
    end
    
    % Save tracklets
    save(sprintf('%s/%s/L1-tracklets/tracklets%d_%s.mat', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        'tracklets');
    
    % Clean up
    clear all_dets appearance detections features frames
    
end
end
