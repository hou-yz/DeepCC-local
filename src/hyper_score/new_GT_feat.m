clear
clc

opts=get_opts();
opts.feature_dir = 'gt_features_ide_basis_train_1fps';
opts.tracklets.window_width = 40;

features=[];
for iCam = 1:8
    tmp = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb')';
    tmp(:,3) = local2global(opts.start_frames(iCam), tmp(:,3));
    pids = unique(tmp(:,2));
    all_pooled_lines = [];
    for i = 1:length(pids)
        pid = pids(i);
        same_pid_lines = tmp(tmp(:,2)==pid,:);
        tracklet_grp = floor(same_pid_lines(:,3)/opts.tracklets.window_width);
        unique_grp = unique(tracklet_grp);
        pooled_lines = zeros(length(unique_grp),259);
        for j = 1:length(unique_grp)
            target_grp = unique_grp(j);
            pooled_lines(j,:) = mean(same_pid_lines(tracklet_grp==target_grp,:),1);
        end
        all_pooled_lines = [all_pooled_lines;pooled_lines];
    end
    
    features   = [features;all_pooled_lines];
end



hdf5write(sprintf('%s/L0-features/%s/tracklet_features.h5',opts.dataset_path,opts.feature_dir),'/emb',features');