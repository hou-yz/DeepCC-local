function compute_L3_identities(opts)
% Computes multi-camera trajectories from single-camera trajectories
hyper_score_param = load(fullfile('src','hyper_score/logs',opts.model_name));

filename = sprintf('%s/%s/L3-identities/L2trajectories.mat',opts.experiment_root, opts.experiment_name);

% consturct traj from L2-result
trajectories = loadL2trajectories(opts);
trajectories = loadTrajectoryFeatures(opts, trajectories);
save(filename,'trajectories');

% load from saved
load(filename);
% set consecutive_icam_martix && reintro_time_matrix
% opts.identities.consecutive_icam_matrix = ones(8);
% opts.identities.reintro_time_matrix = opts.identities.window_width*ones(1,8);

identities = trajectories;

for k = 1:length(identities)
    identities(k).trajectories(1).data(:,end+1) = local2global(opts.start_frames(identities(k).trajectories(1).camera) ,identities(k).trajectories(1).data(:,1));
    identities(k).trajectories(1).startFrame = identities(k).trajectories(1).data(1,9);
    identities(k).startFrame = identities(k).trajectories(1).startFrame;
    identities(k).trajectories(1).endFrame = identities(k).trajectories(1).data(end,9);
    identities(k).endFrame   = identities(k).trajectories(1).endFrame;
    identities(k).iCams = identities(k).trajectories(1).camera;
end
identities = sortStruct(identities,'startFrame');

global_interval = opts.sequence_intervals{opts.sequence};
startFrame = global_interval(1);
endFrame = global_interval(1) + opts.identities.window_width - 1;

while startFrame <= global_interval(end)
    clc; fprintf('Window %d...%d\n', startFrame, endFrame);
    
    identities = linkIdentities(opts, identities, startFrame, endFrame,hyper_score_param);
    
    % advance sliding temporal window
    startFrame = endFrame   - opts.identities.window_width/2;
    endFrame   = startFrame + opts.identities.window_width;
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

% save(sprintf('%s/%s/L3-identities/identities_%s.mat', ...
%         opts.experiment_root, ...
%         opts.experiment_name, ...
%         opts.sequence_names{opts.sequence}), ...
%         'identities', '-v7.3');
    
end