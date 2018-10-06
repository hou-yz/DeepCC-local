function compute_L3_identities(opts)
% Computes multi-camera trajectories from single-camera trajectories

filename = sprintf('%s/%s/L3-identities/L2trajectories.mat',opts.experiment_root, opts.experiment_name);

% load traj from L2-result
% trajectories = loadL2trajectories(opts);
% trajectories = loadTrajectoryFeatures(opts, trajectories);
% save(filename,'trajectories');
% load from saved
load(filename);
identities = trajectories;

for k = 1:length(identities)
    identities(k).trajectories(1).data(:,end+1) = local2global(opts.start_frames(identities(k).trajectories(1).camera) ,identities(k).trajectories(1).data(:,1));
    identities(k).trajectories(1).startFrame = identities(k).trajectories(1).data(1,9);
    identities(k).startFrame = identities(k).trajectories(1).startFrame;
    identities(k).trajectories(1).endFrame = identities(k).trajectories(1).data(end,9);
    identities(k).endFrame   = identities(k).trajectories(1).endFrame;
    identities(k).iCams(1)   = identities(k).trajectories(1).camera;
end
identities = sortStruct(identities,'startFrame');

i = 1;
last_repairedIdentities=[];
while i <= length(identities)
    clc; fprintf('One-Hop link for identity %d\n', i);
    
    [identities,if_updated,repairedIdentities,skip_ind] = linkL3Identities(opts, identities, i);
    
    if ~isempty(repairedIdentities) && ~isempty(last_repairedIdentities) 
    if last_repairedIdentities(1).startFrame==repairedIdentities(1).startFrame && last_repairedIdentities(end).endFrame==repairedIdentities(end).endFrame
        i=i+1;
        continue
    end
    end
    if skip_ind
        i=i+skip_ind;
        continue
    end
    last_repairedIdentities=repairedIdentities;
    
    if if_updated
        continue; % find one-hop for the newly updated traj
    end
    i=i+1;
end
%% save results
fprintf('Saving results\n');
trackerOutputL3 = identities2mat(identities);
for iCam = 1:opts.num_cam
    cam_data = trackerOutputL3(trackerOutputL3(:,1) == iCam,2:end);
    dlmwrite(sprintf('%s/%s/L3-identities/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        cam_data, 'delimiter', ' ', 'precision', 6);
end