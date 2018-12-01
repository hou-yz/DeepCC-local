clc
clear

opts=get_opts();
opts.tracklets.window_width = 40;

% opts.visualize = true;
opts.sequence = 1;
opts.feature_dir = 'gt_features_fc256_1fps_trainBN_crop';
% Computes tracklets for all cameras

fps=60;

newGTs = cellmat(1,8,0,0,0);
spatialGroupID_max = zeros(1,8);
for iCam = 1:8
    
    sequence_window   = opts.sequence_intervals{opts.sequence};
    start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
    end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));
    
    interest_frames = start_frame:round(60/fps):end_frame;
    
    % Load GTs for current camera
    trainData = load(fullfile(opts.dataset_path, 'ground_truth','trainval.mat'));
    trainData = trainData.trainData;
    
    % load feat
    features = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb');
    features = features';
    
    in_time_range_ids = ismember(trainData(:,3),interest_frames) & trainData(:,1)==iCam;
    all_gts   = trainData(in_time_range_ids,2:end);
    in_time_range_ids = ismember(features(:,3),interest_frames) & features(:,2)==iCam;
    all_feat = features(in_time_range_ids,:);
    
    % Compute tracklets for every 1-second interval
    
    node_feats=cellmat(1,8,0,0,0);
    node_ids=cellmat(1,8,0,0,0);
    edges=cellmat(1,8,0,0,0);
    
    for window_start_frame   = start_frame : opts.tracklets.window_width : end_frame
        fprintf('%d/%d\n', window_start_frame, end_frame);
        
        % Retrieve detections in current window
        window_end_frame     = window_start_frame + opts.tracklets.window_width - 1;
        window_frames        = window_start_frame : window_end_frame;
        window_inds          = find(ismember(all_gts(:,2),window_frames));
        if isempty(window_inds)
            continue
        end
        gts_in_window = all_gts(window_inds,:);
        feat_in_window = all_feat(window_inds,4:end);
        
        gts_in_window(:,7:end) = [];
        gts_in_window(:,[1 2]) = gts_in_window(:,[2 1]);
        filteredGTs = gts_in_window;
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        [edge_weight,spatialGroupID_max(iCam)] = GT_L1_processing(opts, filteredGTs, window_start_frame, window_end_frame,spatialGroupID_max(iCam),0);
%         edges,feat_in_window
        node_ids{spatialGroupID_max(iCam)}=gts_in_window(:,2);
        node_feats{spatialGroupID_max(iCam)}=feat_in_window;
        edges{spatialGroupID_max(iCam)}=edge_weight;
    end
    save(fullfile(opts.dataset_path, 'ground_truth','GCN_L1',sprintf('node_ids%d.mat',iCam)),'node_ids');
    save(fullfile(opts.dataset_path, 'ground_truth','GCN_L1',sprintf('node_feats%d.mat',iCam)),'node_feats');
    save(fullfile(opts.dataset_path, 'ground_truth','GCN_L1',sprintf('edges%d.mat',iCam)),'edges');
end

% hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_L1_%s.h5',opts.sequence_names{opts.sequence})), '/hyperGT',res');


