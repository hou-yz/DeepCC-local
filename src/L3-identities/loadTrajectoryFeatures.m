function trajectories = loadTrajectoryFeatures(opts, scene, trajectories)
count = 1;
if opts.dataset == 0
    cam_pool = 1:opts.num_cam;
elseif opts.dataset == 2
    cam_pool = opts.cams_in_scene{scene};
end
for iCam = cam_pool
    traj_for_iCam = load(fullfile(opts.experiment_root, opts.experiment_name, 'L2-trajectories', sprintf('trajectories%d_%s.mat',iCam, opts.sequence_names{opts.sequence})));
    removed_ids = traj_for_iCam.removedIDs;
    traj_for_iCam = traj_for_iCam.trajectories;
    for id = 1:length(traj_for_iCam)
        if ismember(id,removed_ids)
            continue
        end
        trajectory_feature = traj_for_iCam(id).feature;
        
        % Trajectory feature is the average of all features (if more than one image)
        trajectories(count).trajectories(1).feature = trajectory_feature;
        count = count + 1;
        
    end
end
end

