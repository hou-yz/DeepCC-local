function compute_L2_trajectories(opts)
% Computes single-camera trajectories from tracklets
for iCam = 1:8

    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    trajectoriesFromTracklets = trackletsToTrajectories(tracklets,1:length(tracklets));
    
    opts.current_camera = iCam;
    sequence_interval = opts.sequence_intervals{opts.sequence};
    startFrame = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) - opts.trajectories.window_width);
    endFrame   = global2local(opts.start_frames(opts.current_camera), sequence_interval(1) + opts.trajectories.window_width);

    trajectories = trajectoriesFromTracklets; 
    
    while startFrame <= global2local(opts.start_frames(opts.current_camera), sequence_interval(end))
        % Display loop state
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, startFrame, endFrame);

        % Compute trajectories in current time window
        trajectories = createTrajectories( opts, trajectories, startFrame, endFrame);

        % Update loop range
        startFrame = endFrame   - opts.trajectories.overlap;
        endFrame   = startFrame + opts.trajectories.window_width;
    end

    % Convert trajectories 
    trackerOutputRaw = trajectoriesToTop(trajectories);
    % Interpolate missing detections
    trackerOutputFilled = fillTrajectories(trackerOutputRaw);
    % Remove spurius tracks
    trackerOutputRemoved = removeShortTracks(trackerOutputFilled, opts.minimum_trajectory_length);
    % Make identities 1-indexed
    [~, ~, ic] = unique(trackerOutputRemoved(:,2));
    trackerOutputRemoved(:,2) = ic;
    trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

    %% Save trajectories
    fprintf('Saving results\n');
    fileOutput = trackerOutput(:, [1:6]);
    dlmwrite(sprintf('%s/%s/L2-trajectories/cam%d_%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        iCam, ...
        opts.sequence_names{opts.sequence}), ...
        fileOutput, 'delimiter', ' ', 'precision', 6);


end
end