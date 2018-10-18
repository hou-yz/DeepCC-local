clc
clear

opts=get_opts();

% opts.visualize = true;
opts.sequence = 5;
opts.feature_dir = 'gt_features_fc256_1fps_trainBN_crop';
% Computes tracklets for all cameras

    

newGTs = cellmat(1,8,0,0,0);
spatialGroupID_max = zeros(1,8);
parfor iCam = 1:8
    
    sequence_window   = opts.sequence_intervals{opts.sequence};
    start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
    end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));
    
    % Load GTs for current camera
    trainData = load(fullfile(opts.dataset_path, 'ground_truth','trainval.mat'));
    trainData = trainData.trainData;
    
    % load feat
    features = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb');
    features = features';
    
    in_time_range_ids = trainData(:,3)>=start_frame & trainData(:,3)<=end_frame & trainData(:,1)==iCam;
    all_gts   = trainData(in_time_range_ids,2:end);
    in_time_range_ids =( features(:,3)>=start_frame & features(:,3)<=end_frame & features(:,2)==iCam),
    all_feat = features(in_time_range_ids,:);
    
    % Compute tracklets for every 1-second interval
    
    newGT=[];
    
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        window_inds          = find(ismember(all_gts(:,2),window_frames));
        gts_in_window = all_gts(window_inds,:);
        feat_in_window = all_feat(window_inds,4:end);
        
        gts_in_window(:,7:end) = [];
        gts_in_window(:,[1 2]) = gts_in_window(:,[2 1]);
        filteredGTs = gts_in_window;
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        [appendedGTs,spatialGroupID_max(iCam)] = GT_processing(opts, filteredGTs, window_start_frame, window_end_frame,spatialGroupID_max(iCam),0);
        appendedGTs = [appendedGTs(),zeros(size(window_inds)),feat_in_window];
        newGT = [newGT;appendedGTs];
    end
    newGT(:,[1 2]) = newGT(:,[2 1]);
    newGT = [ones(size(newGT,1),1)*iCam,newGT];
%     newGT = sort(newGT);
    newGTs{iCam} = [newGTs{iCam};newGT];
end

res = [];
for iCam = 1:8
    newGT = newGTs{iCam};
    newGT(:,4) = newGT(:,4)+sum(spatialGroupID_max(1:iCam-1));
    res = [res;newGT];
%     hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_%s_%d.h5',opts.sequence_names{opts.sequence},iCam)), '/hyperGT',newGTs{iCam}');
end

hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_%s.h5',opts.sequence_names{opts.sequence})), '/hyperGT',res');


