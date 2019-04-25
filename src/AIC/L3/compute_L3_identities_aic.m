function compute_L3_identities_aic(opts)
    % Computes multi-camera trajectories from single-camera trajectories
    if opts.identities.og_appear_score
        appear_model_param = [];
    else
        appear_model_param = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
    end
    if opts.identities.og_motion_score
        motion_model_param = [];
    else
        motion_model_param = load(fullfile('src','hyper_score/logs',opts.motion_model_name));
    end

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
        iCam = identities(k).trajectories.camera;
        frame_offset = opts.frame_offset{opts.trainval_scene_by_icam(iCam)};
        identities(k).trajectories(1).data(:,end+1) = local2global(frame_offset(iCam), identities(k).trajectories(1).data(:,1));
        identities(k).trajectories(1).startFrame = identities(k).trajectories(1).data(1,9);
        identities(k).startFrame = identities(k).trajectories(1).startFrame;
        identities(k).trajectories(1).endFrame = identities(k).trajectories(1).data(end,9);
        identities(k).endFrame   = identities(k).trajectories(1).endFrame;
        identities(k).iCams = identities(k).trajectories(1).camera;
    end
    identities = sortStruct(identities,'startFrame');        

    startFrame     = 1;
    endFrame       = 1 + opts.identities.window_width - 1;

    while startFrame <= 6000
        clc; fprintf('Window %d...%d\n', startFrame, endFrame);

        identities = linkIdentities(opts, identities, startFrame, endFrame,appear_model_param,motion_model_param);

        % advance sliding temporal window
        startFrame = endFrame   - opts.identities.window_width/2;
        endFrame   = startFrame + opts.identities.window_width;
    end
    
%     to_delete = [];
%     
%     for index = 1:length(identities)
%         if length(identities(index).iCams) > 8
%             to_delete = [to_delete, index];
%         end
%     end
%     
%     identities(to_delete) = [];
    
    %% save results
    fprintf('Saving results\n');
    trackerOutputL3 = identities2mat(identities);
    for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
        cam_data = trackerOutputL3(trackerOutputL3(:,1) == iCam,2:end);
        dlmwrite(sprintf('%s/%s/L3-identities/cam%d_%s.txt', ...
            opts.experiment_root, ...
            opts.experiment_name, ...
            iCam, ...
            opts.sequence_names{opts.sequence}), ...
            cam_data, 'delimiter', ' ', 'precision', 6);
    end
    end

    % save(sprintf('%s/%s/L3-identities/identities_%s.mat', ...
    %         opts.experiment_root, ...
    %         opts.experiment_name, ...
    %         opts.sequence_names{opts.sequence}), ...
    %         'identities');
    
end