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

    all_scenario_ids = [];
    for scene = opts.seqs{opts.sequence}
    opts.current_scene = scene;
    % consturct traj from L2-result
    trajectories = loadL2trajectories(opts, scene);
    identities = trajectories;    
    for k = 1:length(identities)
        iCam = identities(k).trajectories.camera;
        frame_offset = opts.frame_offset{opts.current_scene};
        identities(k).trajectories(1).data(:,end+1) = local2global(frame_offset(iCam), identities(k).trajectories(1).data(:,1));
        identities(k).trajectories(1).startFrame = identities(k).trajectories(1).data(1,9);
        identities(k).startFrame = identities(k).trajectories(1).startFrame;
        identities(k).trajectories(1).endFrame = identities(k).trajectories(1).data(end,9);
        identities(k).endFrame   = identities(k).trajectories(1).endFrame;
        identities(k).iCams = identities(k).trajectories(1).camera;
    end
    identities = sortStruct(identities,'startFrame');        
    
    %% subscenes
    for cams_in_subscene = opts.subscenes{scene}
        
    if ismember(cams_in_subscene{1},10)
        window_width = 1200;
    else
        window_width = 500;
    end
        
    startFrame     = 0;
    endFrame       = window_width;
    while startFrame <= 6000
        clc; fprintf('Window %d...%d\n', startFrame, endFrame);

        identities = linkIdentities_subscene(opts, identities, startFrame, endFrame,appear_model_param,motion_model_param,cams_in_subscene{1});

        % advance sliding temporal window
        startFrame = startFrame + window_width/2;
        endFrame   = startFrame + window_width;
    end
    end
    
    %% all cams in scene
    if scene <= 2
        window_width = 500;
    else
        window_width = 2400;
    end
    
    startFrame     = 0;
    endFrame       = window_width;
    while startFrame <= 6000
        clc; fprintf('Window %d...%d\n', startFrame, endFrame);

        identities = linkIdentities_aic(opts, identities, startFrame, endFrame,appear_model_param,motion_model_param);

        % advance sliding temporal window
        startFrame = startFrame + window_width/2;
        endFrame   = startFrame + window_width;
    end
    
    %% remove single-camera ones
%     to_delete = [];
%     for index = 1:length(identities)
%         if length(unique(identities(index).iCams)) == 1
%             to_delete = [to_delete, index];
%         end
%     end
%     identities(to_delete) = [];
    
    all_scenario_ids = [all_scenario_ids,identities];
    end
    
    %% save results
    fprintf('Saving results\n');
    trackerOutputL3 = identities2mat(all_scenario_ids);
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
end