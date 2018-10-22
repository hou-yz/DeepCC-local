clc
clear

opts=get_opts();

% opts.visualize = true;
opts.sequence = 2;
opts.experiment_name = '1fps_L1L2';
opts.feature_dir = 'det_features_fc256_1fps_trainBN_crop_trainval_mini';

newGTs = cellmat(1,8,0,0,0);
spatialGroupID_max = zeros(1,8);
% Computes single-camera trajectories from tracklets
for iCam = 1:8
    sequence_window   = opts.sequence_intervals{opts.sequence};
    start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
    end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));
    
    filename =sprintf('%s/ground_truth/tracklets%d_%s.mat',opts.dataset_path,iCam,opts.sequence_names{opts.sequence});
    if ~exist(filename, 'file')
    % Load GTs for current camera
    trainData = load(fullfile(opts.dataset_path, 'ground_truth','trainval.mat'));
    trainData = trainData.trainData;
    
    in_time_range_ids = trainData(:,3)>=start_frame & trainData(:,3)<=end_frame & trainData(:,1)==iCam;
    all_gts   = trainData(in_time_range_ids,2:7);
    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    fields={'center','centerWorld','features','realdata','mask','interval','segmentStart','segmentInterval','segmentEnd','ids'};
    tracklets = rmfield(tracklets,fields);
    for i = 1:length(tracklets)
        window_frames = tracklets(i).data(:,1);
        gts_in_window = all_gts(ismember(all_gts(:,2),window_frames),:);
        gts_in_window(:,[1 2]) = gts_in_window(:,[2 1]);
        pids = unique(gts_in_window(:,2));
        IoUs = zeros(length(window_frames),length(pids));
        for k = 1:length(window_frames)
            frame = window_frames(k);
            bbox_det = tracklets(i).data(k,3:6);
            for j = 1:length(pids)
                pid = pids(j);
                index = find((gts_in_window(:,2)==pid) .* (gts_in_window(:,1)==frame));
                bbox_gt = gts_in_window(index,3:6);
                if isempty(bbox_gt)
                    IoUs(k,j) = 0;
                else
                    IoUs(k,j) = bboxOverlapRatio(bbox_det,bbox_gt);
                end
            end
        end
        IoUs=mean(IoUs,1);
        [IoU,j]=max(IoUs);
        pid = pids(j);
        if IoU<0.2
            pid=-1;
        end
        if isempty(pids)
            pid=-1;
        end
        tracklets(i).id=pid;
        if isempty(tracklets(i).id)
            i
        end
    end
    
    % Save trajectories
    save(filename, 'tracklets');
    else
        load(filename)
        [~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);
        centerFrame     = round(mean(intervals,2));
        centers         = 0.5 * (endpoint + startpoint);
        pids = [tracklets.id]';
        feat = reshape([tracklets.feature]',256,[])';
        newGTs{iCam} = [ones(size(pids))*iCam,pids,centerFrame,zeros(size(pids,1),2),centers,velocity,feat];
        
        for i=1:ceil((end_frame-start_frame+1)/opts.trajectories.window_width)
        % Display loop state
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, start_frame+(i-1)*opts.trajectories.window_width, start_frame+i*opts.trajectories.window_width);
        
        indexs = logical((newGTs{iCam}(:,3)>=(start_frame+(i-1)*opts.trajectories.window_width)) .*(newGTs{iCam}(:,3)<(start_frame+i*opts.trajectories.window_width)));
        
        spatialGroupID = 1+spatialGroupID_max(iCam);
        spatialGroupID_max(iCam) = spatialGroupID;
        newGTs{iCam}(indexs,4)=spatialGroupID;
        end
    end
end

res = [];
for iCam = 1:8
    newGT = newGTs{iCam};
    newGT(:,4) = newGT(:,4)+sum(spatialGroupID_max(1:iCam-1));
    res = [res;newGT];
%     hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_%s_%d.h5',opts.sequence_names{opts.sequence},iCam)), '/hyperGT',newGTs{iCam}');
end

hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_%s.h5',opts.sequence_names{opts.sequence})), '/hyperGT',res');