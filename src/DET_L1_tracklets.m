clc
clear

opts=get_opts();
opts.tracklets.window_width = 40;

% opts.visualize = true;
opts.sequence = 8;
opts.feature_dir = 'det_features_fc256_train_1fps_trainBN_trainval';
% Computes tracklets for all cameras

mkdir(fullfile(opts.dataset_path, 'ground_truth','GCN_L1_det'))
mkdir(fullfile(opts.dataset_path, 'ground_truth','GCN_L1_det',opts.sequence_names{opts.sequence}))

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
    
    %load Det for current camera
    detections = load(fullfile(opts.dataset_path, 'detections','OpenPose', sprintf('camera%d.mat',iCam)));
    detections = detections.detections;
    
    in_time_range_ids = ismember(trainData(:,3),interest_frames) & trainData(:,1)==iCam;
    all_gts   = trainData(in_time_range_ids,2:end);
    
    % Load features for all detections
    % features   = h5read(sprintf('%s/%s/L0-features/features%d.h5',opts.experiment_root,opts.experiment_name,iCam),'/emb');
    if isempty(opts.feature_dir)
        features   = h5read(sprintf('%s/L0-features/features%d.h5',opts.dataset_path,iCam),'/emb');
    else
        features   = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb');
    end
    in_time_range_ids = detections(:,2)>=start_frame & detections(:,2)<=end_frame;
    all_dets   = detections(in_time_range_ids,:);
    if size(features,2) == size(detections,1)
        features = features(:,in_time_range_ids);
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
        filteredFeatures = cell2mat(appearance(window_inds(valid)));
        
        
        % Retrieve GTs in current window
        window_inds          = find(ismember(all_gts(:,2),window_frames));
        if isempty(window_inds)
            continue
        end
        gts_in_window = all_gts(window_inds,:);
        
        gts_in_window(:,7:end) = [];
        gts_in_window(:,[1 2]) = gts_in_window(:,[2 1]);
        filteredGTs = gts_in_window;
        
        % Compute tracklets in current window
        % Then add them to the list of all tracklets
        [edge_weight,spatialGroupID_max,node_ids,det_to_remove_indexs] = DET_L1_processing(opts, filteredGTs, filteredDetections, window_start_frame, window_end_frame,spatialGroupID_max);
%         edges,feat_in_window
        node_ids_r = repmat(node_ids,1,length(node_ids));
        node_ids_c = repmat(node_ids',length(node_ids),1);
        edge_target=node_ids_r==node_ids_c;
        edge_target(node_ids_c<=0 + node_ids_r<=0)=0;
        node_feat=filteredFeatures;
        node_feat(det_to_remove_indexs,:)=[];
        edge_iou=edge_weight;
        if size(edge_iou,1)~=size(node_ids,1)
            size(edge_iou,1)
        end
        
        if isempty(node_ids)
            continue
        end
save(fullfile(opts.dataset_path, 'ground_truth','GCN_L1_det',opts.sequence_names{opts.sequence},sprintf('data_%06d.mat',spatialGroupID_max)),'edge_target','node_feat','edge_iou','-v7.3');
    end
end
% hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_L1_%s.h5',opts.sequence_names{opts.sequence})), '/hyperGT',res');


