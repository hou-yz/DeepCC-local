function compute_L1_tracklets_mot(opts)
% Computes tracklets for all cameras

appear_model_param = [];
motion_model_param = [];
for i = 1:length(opts.seqs)
    iCam = opts.seqs(i);
    opts.current_camera = iCam;
    seq_name = sprintf('MOT16-%02d',iCam);
    
    % Load MOT detections for current camera
    detections = importdata(fullfile(opts.dataset_path, 'train',seq_name,'det','det.txt'));
    
    % Load features for all detections
    features = h5read(sprintf('%s/all_seq_feat.h5',opts.feature_dir),'/emb')';
    features = features(features(:,1)==iCam,4:end);
    
    appearance = cell(size(detections,1),1);
    frames     = cell(size(detections,1),1);
    for k = 1:length(frames)
        appearance{k} = double(features(k,:));
        frames{k} = detections(k,1);
    end
    
    start_frame = min(detections(:,1));
    end_frame = max(detections(:,1));
    
    % Compute tracklets for every 1-second interval
    tracklets = struct([]);
    
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        window_inds          = find(ismember(detections(:,1),window_frames));
        detections_in_window = detections(window_inds,:);
        detections_conf      = detections_in_window(:,7);
        
        % Use only valid detections
        valid                         = detections_conf>opts.render_threshold;
        detections_in_window          = detections_in_window(detections_conf>opts.render_threshold,:);
        detections_in_window(:,7:end) = [];
        detections_in_window(:,[2 1]) = [ones(size(detections_in_window,1),1)*iCam,detections_in_window(:,1)];
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
