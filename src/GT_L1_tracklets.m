clc
clear

opts=get_opts();

opts.sequence = 1;
% Computes tracklets for all cameras

newGTs = [];
for iCam = 1:8
    
    opts.current_camera = iCam;
    
    sequence_window   = opts.sequence_intervals{opts.sequence};
    start_frame       = global2local(opts.start_frames(opts.current_camera), sequence_window(1));
    end_frame         = global2local(opts.start_frames(opts.current_camera), sequence_window(end));
    
    % Load OpenPose detections for current camera
    load(fullfile(opts.dataset_path, 'ground_truth','trainval.mat'));
    
    in_time_range_ids = trainData(:,3)>=start_frame & trainData(:,3)<=end_frame & trainData(:,1)==iCam;
    all_gts   = trainData(in_time_range_ids,2:end);
    appearance = cell(size(all_gts,1),1);
    frames     = cell(size(all_gts,1),1);
    for k = 1:length(frames)
        frames{k} = trainData(k,3);
    end
    
    % Compute tracklets for every 1-second interval
    spatialGroupID_max = 0;
    newGT=[];
    
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        window_inds          = find(ismember(all_gts(:,2),window_frames));
        gts_in_window = all_gts(window_inds,:);
        
        gts_in_window(:,7:end) = [];
        gts_in_window(:,[1 2]) = gts_in_window(:,[2 1]);
        filteredGTs = gts_in_window;
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        [appendedGTs,spatialGroupID_max] = GT_processing(opts, filteredGTs, window_start_frame, window_end_frame,spatialGroupID_max);
        newGT = [newGT;appendedGTs];
    end
    newGT(:,[1 2]) = newGT(:,[2 1]);
    newGT = [ones(size(newGT,1),1)*iCam,newGT];
    newGT = sort(newGT);
    newGTs = [newGTs;newGT];
end

save(fullfile(opts.dataset_path, 'ground_truth','newGTs.mat'), 'newGTs');
