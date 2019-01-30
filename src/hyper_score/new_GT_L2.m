clear
clc

opts=get_opts();
opts.sequence = 7;
opts.feature_dir = 'gt_features_fc256_train_1fps_trainBN';

trainData = load(fullfile(opts.dataset_path, 'ground_truth','trainval.mat'));
trainData = trainData.trainData;
features=[];
for iCam = 1:8
    line_id = find(trainData(:,1)==iCam);
    trainData(line_id==1,3) = local2global(opts.start_frames(iCam), trainData(line_id==1,3));
    features   = [features;h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb')'];
end
trainData = trainData(ismember(trainData(:,3),opts.sequence_intervals{opts.sequence}),:);

opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
% [camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY]

pids = unique(trainData(:,2));
tracks =  struct([]);
num_traj = 0;
for i = 1:length(pids)
    pid = pids(i);
    same_pid_line_ids = find(trainData(:,2) == pid);
    same_pid_GTs = trainData(same_pid_line_ids,:);
    same_pid_GTs = sortrows(same_pid_GTs,3);
    
    switch_cam_lines_id = [1;(same_pid_GTs(1:end-1,1)==same_pid_GTs(2:end,1))==0];
    % also add person_id who leave and enter the same camera after a while 
    switch_cam_lines_id(([0;(same_pid_GTs(1:end-1,3)+opts.trajectories.window_width<same_pid_GTs(2:end,3)).*(same_pid_GTs(1:end-1,1)==same_pid_GTs(2:end,1))])==1)=1;

    intro_lines_id = find(switch_cam_lines_id==1);
    outro_lines_id = [intro_lines_id(2:end)-1;length(same_pid_line_ids)];
    trajectories = [];
    trajectory = struct('tracklets',[],'startFrame',inf,'endFrame',-inf,'data',[],'worldPos',[],'features',[]);
    for j = 1:length(intro_lines_id)
        trajectory.pid = pid;
        trajectory.data = same_pid_GTs(intro_lines_id(j):outro_lines_id(j),:);
        trajectory.iCam = unique(trajectory.data(:,1));
        trajectory.startFrame = trajectory.data(1,3);
        trajectory.endFrame = trajectory.data(end,3);
        trajectory.worldPos = trajectory.data(:,[8,9]);
        trajectory.frame = trajectory.data(:,3);
        tracklets = struct([]);
        time_range_start = trajectory.startFrame:opts.tracklets.window_width:trajectory.endFrame-1;
        time_range_end   = trajectory.startFrame+opts.tracklets.window_width-1:opts.tracklets.window_width:trajectory.endFrame;
        if length(time_range_end)<length(time_range_start)
            time_range_end = [time_range_end,trajectory.endFrame];
        end
        
        for k = 1:length(time_range_start)
            time_range = time_range_start(k):time_range_end(k);
            target_line_ids = find(ismember(trajectory.data(:,3),time_range));
            if isempty(target_line_ids)
                continue
            end
            tracklets(end+1).pid = pid;
            tracklets(end).iCam = trajectory.iCam;
            tracklets(end).data = trajectory.data(target_line_ids,:);
            tracklets(end).startFrame = tracklets(end).data(1,3);
            tracklets(end).endFrame = tracklets(end).data(end,3);
            tracklets(end).worldPos = tracklets(end).data(:,[8,9]);
            
        end
        trajectory.tracklets = tracklets;
        trajectories = [trajectories; trajectory];
        num_traj = num_traj + 1;
    end
    tracks(end+1).pid = pid;
    tracks(end).trajectories = trajectories;
    tracks(end).startFrame = min(same_pid_GTs(:,3));
    tracks(end).endFrame = max(same_pid_GTs(:,3));
end

output = cellmat(3,num_traj,0,0,0);
index_traj = 0;
for i = 1:length(pids)
    trajectories = tracks(i).trajectories;
    for j = 1:length(trajectories)
        target = zeros(1,3);
        target(2) = trajectories(j).iCam;
        if j == 1
            target(1) = -1;
        else
            target(1) = trajectories(j-1).iCam;
        end
        if j == length(trajectories)
            target(3) = -1;
        else
            target(3) = trajectories(j+1).iCam; 
        end
        index_traj = index_traj+1;
        output{1,index_traj} = target;
        output{2,index_traj} = trajectories(j).worldPos;
        output{3,index_traj} = trajectories(j).frame;
    end
end

fpath = fullfile(opts.dataset_path, 'ground_truth',sprintf('GT_L3_motion_%s.mat',opts.sequence_names{opts.sequence}));
save(fpath, 'output');


