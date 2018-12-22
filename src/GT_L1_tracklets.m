clc
clear

opts=get_opts();
opts.tracklets.window_width = 40;

% opts.visualize = true;
opts.sequence = 7;
opts.feature_dir = 'gt_features_fc256_1fps_trainBN_crop';
% Computes tracklets for all cameras

mkdir(fullfile(opts.dataset_path, 'ground_truth','GCN_L1'))
mkdir(fullfile(opts.dataset_path, 'ground_truth','GCN_L1',opts.sequence_names{opts.sequence}))

fps=60;

newGTs = cellmat(1,8,0,0,0);
spatialGroupID_max = 0;
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
        [edge_weight,spatialGroupID_max] = GT_L1_processing(opts, filteredGTs, window_start_frame, window_end_frame,spatialGroupID_max,0);
%         edges,feat_in_window
        node_ids=gts_in_window(:,2);
        node_ids_r = repmat(node_ids,1,length(node_ids));
        node_ids_c = repmat(node_ids',length(node_ids),1);
        edge_target=node_ids_r==node_ids_c;
        node_feat=feat_in_window;
        edge_iou=edge_weight;
        
save(fullfile(opts.dataset_path, 'ground_truth','GCN_L1',opts.sequence_names{opts.sequence},sprintf('data_%06d.mat',spatialGroupID_max)),'edge_target','node_feat','edge_iou','-v7.3');
    end
end
% hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_L1_%s.h5',opts.sequence_names{opts.sequence})), '/hyperGT',res');


