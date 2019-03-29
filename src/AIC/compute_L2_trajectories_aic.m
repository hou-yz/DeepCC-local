function compute_L2_trajectories_aic(opts)
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
for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
    opts.current_camera = iCam;

    % Load OpenPose detections for current camera
    detections      = load(sprintf('%s/train/S%02d/c%03d/det/det_yolo3.txt', opts.dataset_path, scene, iCam));
    startFrame     = detections(1, 1);
    endFrame       = startFrame + opts.trajectories.window_width - 1;
%     endFrame       = detections(end, 1);

    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    if opts.fft
        tracklets = fft_tracklet_feat(opts, tracklets);
    end

    trajectoriesFromTracklets = trackletsToTrajectories(opts, tracklets,1:length(tracklets));

    opts.current_camera = iCam;

    trajectories = trajectoriesFromTracklets;

    while startFrame <= detections(end, 1)
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

    if iCam == 13
        trackerOutput = fps_8to10(trackerOutput);
    end

    
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
end