function compute_L2_trajectories_mot(opts)
% Computes single-camera trajectories from tracklets

appear_model_param = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
motion_model_param = [];
for i = 1:length(opts.seqs)
    iCam = opts.seqs(i);
    opts.current_camera = iCam;
    seq_name = sprintf('MOT16-%02d',iCam);
    % Load MOT detections for current camera
    detections = importdata(fullfile(opts.dataset_path, 'train',seq_name,'det','det.txt'));
    
    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));

    trajectoriesFromTracklets = trackletsToTrajectories(opts, tracklets,1:length(tracklets));
    
    startFrame = min(detections(:,1));
    endFrame = min(detections(:,1)) + opts.trajectories.window_width - 1;

    trajectories = trajectoriesFromTracklets; 
    
    while startFrame <= max(detections(:,1))
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
    dlmwrite(sprintf('%s/%s/L2-trajectories/%s.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        seq_name), ...
        fileOutput, 'delimiter', ' ', 'precision', 6);

end
end