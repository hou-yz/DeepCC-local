function compute_L4_identities(opts)
% Computes multi-camera trajectories from L3 one hop ids
% load L3 one-hop ids
load(fullfile(opts.experiment_root, opts.experiment_name, 'L3-identities', sprintf('identities_%s.mat',opts.sequence_names{opts.sequence})));
% set consecutive_icam_martix && reintro_time_matrix
opts.identities.consecutive_icam_matrix = ones(8);
opts.identities.reintro_time_matrix = opts.identities.window_width*ones(1,8);


global_interval = opts.sequence_intervals{opts.sequence};
startFrame = global_interval(1);
endFrame = global_interval(1) + opts.identities.window_width - 1;

while startFrame <= global_interval(end)
    clc; fprintf('Window %d...%d\n', startFrame, endFrame);
    
    identities = linkIdentities(opts, identities, startFrame, endFrame);
    
    % advance sliding temporal window
    startFrame = endFrame   - opts.identities.window_width/2;
    endFrame   = startFrame + opts.identities.window_width;
end
%% save results
fprintf('Saving results\n');
trackerOutputL4 = identities2mat(identities);
for iCam = 1:opts.num_cam
    cam_data = trackerOutputL4(trackerOutputL4(:,1) == iCam,2:end);
    dlmwrite(sprintf('%s/%s/L4-identities/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        cam_data, 'delimiter', ' ', 'precision', 6);
end