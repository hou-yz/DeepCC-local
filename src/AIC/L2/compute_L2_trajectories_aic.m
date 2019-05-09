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
    opts.current_scene = scene;
    for i = 1:length(opts.cams_in_scene{scene})
    iCam = opts.cams_in_scene{scene}(i);
    opts.current_camera = iCam;

    % Load OpenPose detections for current camera
    detections      = load(sprintf('%s/%s/S%02d/c%03d/det/det_%s.txt', opts.dataset_path, opts.sub_dir{opts.sequence}, scene, iCam, opts.detections));
    startFrame     = detections(1, 1);
    endFrame       = startFrame + opts.trajectories.window_width - 1;
%     endFrame       = detections(end, 1);

    % Initialize
    tracklets = load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    tracklets = tracklets.tracklets;
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
        trajectories = createTrajectories_aic(opts, trajectories, startFrame, endFrame, iCam,appear_model_param,motion_model_param);

        % Update loop range
        startFrame = endFrame   - opts.trajectories.window_width/2;
        endFrame   = startFrame + opts.trajectories.window_width;
    end
    
    % Convert trajectories
    trackerOutputRaw = trajectoriesToTop(trajectories);
    % Remove spurius tracks
    [trackerOutputRemoved, removedIDs] = removeShortTracks(trackerOutputRaw, opts.minimum_trajectory_length);
    trajectories(removedIDs) = [];
    % Make identities 1-indexed
    [~, ~, ic] = unique(trackerOutputRemoved(:,2));
    trackerOutputRemoved(:,2) = ic;
    trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

    %% Save output
    fprintf('Saving results\n');
    fileOutput = trackerOutput;
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
        'trajectories', '-v7.3');


    end
end
end