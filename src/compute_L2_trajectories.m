function compute_L2_trajectories(opts)
% Computes single-camera trajectories from tracklets

if opts.trajectories.og_appear_score
    appear_model_param = [];
else
    appear_model_param = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
end
if opts.trajectories.og_motion_score
    motion_model_param = [];
else
    motion_model_param = load(fullfile('src','hyper_score/logs',opts.motion_model_name));
end
for iCam = 1:8

    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    if opts.fft
        tracklets = fft_tracklet_feat(opts, tracklets);
    end

    trajectoriesFromTracklets = trackletsToTrajectories(opts, tracklets,1:length(tracklets));
    
    opts.current_camera = iCam;
    sequence_interval = opts.sequence_intervals{opts.sequence};
    startFrame = global2local(opts.start_frames(opts.current_camera), sequence_interval(1));
    endFrame   = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) + opts.trajectories.window_width - 1);

    trajectories = trajectoriesFromTracklets; 
    
    while startFrame <= global2local(opts.start_frames(opts.current_camera), sequence_interval(end))
        % Display loop state
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, startFrame, endFrame);

        % Compute trajectories in current time window
        trajectories = createTrajectories(opts, trajectories, startFrame, endFrame, iCam,appear_model_param,motion_model_param);

        % Update loop range
        startFrame = endFrame   - opts.trajectories.window_width/2;
        endFrame   = startFrame + opts.trajectories.window_width;
    end

    % Convert trajectories 
    trackerOutputRaw = trajectoriesToTop(trajectories);
    % Interpolate missing detections
    trackerOutputFilled = fillTrajectories(trackerOutputRaw);
    % Remove spurius tracks
    [trackerOutputRemoved, removedIDs] = removeShortTracks(trackerOutputFilled, opts.minimum_trajectory_length);
    % Make identities 1-indexed
    [~, ~, ic] = unique(trackerOutputRemoved(:,2));
    trackerOutputRemoved(:,2) = ic;
    trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

    %% Save output
    fprintf('Saving results\n');
    fileOutput = trackerOutput(:, [1:6]);
    dlmwrite(sprintf('%s/%s/L2-trajectories/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        fileOutput, 'delimiter', ' ', 'precision', 6);
    
    % Save trajectories
    save(sprintf('%s/%s/L2-trajectories/trajectories%d_%s.mat', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        'trajectories', 'removedIDs');


end
end