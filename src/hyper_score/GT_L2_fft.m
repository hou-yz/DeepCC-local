clc
clear

opts=get_opts();

opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
L2_speed = 'mid';
% L2_speed = 'head-tail';

% opts.visualize = true;
opts.sequence = 7;
opts.experiment_name = '1fps_train_IDE_40';

newGTs = cellmat(1,8,0,0,0);
spatialGroupID_max = zeros(1,8);
% Computes single-camera trajectories from tracklets
for iCam = 1:8
    sequence_window   = opts.sequence_intervals{1};
    start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
    end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));
    
    filename = sprintf('%s/ground_truth/%s/tracklets%d_trainval.mat',opts.dataset_path,opts.experiment_name,iCam);
    load(filename)
    pids = [tracklets.id]';
    all_fft_feats = zeros(length(tracklets),1024);
    freq_energy=0;
    tracklets = fft_tracklet_feat(opts, tracklets);
    
        [~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);
        centerFrame     = local2global(opts.start_frames(iCam),round(mean(intervals,2)));
        centers         = 0.5 * (endpoint + startpoint);
        newGTs{iCam} = [ones(size(pids))*iCam,pids,centerFrame,zeros(size(pids,1),1),centers,velocity,zeros(size(pids,1),1),all_fft_feats];
        in_time_range_ids = ismember(centerFrame,opts.sequence_intervals{opts.sequence});
        newGTs{iCam} = newGTs{iCam}(in_time_range_ids,:);
        
    for i=1:ceil(length(sequence_window)/opts.trajectories.window_width)
        % Display loop state
        window = (sequence_window(1)+(i-1)*opts.trajectories.window_width): (sequence_window(1)+i*opts.trajectories.window_width-1);
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, window(1),window(end));
        indexs = ismember(newGTs{iCam}(:,3),window);
        spatialGroupID = 1+spatialGroupID_max(iCam);
        spatialGroupID_max(iCam) = spatialGroupID;
        newGTs{iCam}(indexs,4)=spatialGroupID;
    end
end
res = [];
for iCam = 1:8
    newGT = newGTs{iCam};
    newGT(:,4) = newGT(:,4)+sum(spatialGroupID_max(1:iCam-1));
    if opts.tracklets.window_width == inf
        newGT(:,4) = 0;
    end
    res = [res;newGT];
%     hdf5write(fullfile(opts.dataset_path, 'ground_truth',sprintf('hyperGT_%s_%d.h5',opts.sequence_names{opts.sequence},iCam)), '/hyperGT',newGTs{iCam}');
end


% res(:,4) = 0;
hdf5write(fullfile(opts.dataset_path, 'ground_truth',opts.experiment_name,sprintf('hyperGT_L2_fft_%s_%d.h5',opts.sequence_names{opts.sequence},opts.trajectories.window_width)), '/hyperGT',res');
