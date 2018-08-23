function trajectories = loadL2trajectories(opts)
% Loads single camera trajectories that have been written to disk
trajectories = [];
count = 1;
for iCam = 1:opts.num_cam
    tracker_output = dlmread(fullfile(opts.experiment_root, opts.experiment_name, 'L2-trajectories', sprintf('cam%d_%s.txt',iCam, opts.sequence_names{opts.sequence})));
    
    ids = unique(tracker_output(:,2));
    
    for idx = 1:length(ids)
        id = ids(idx);
        trajectory = [];
        trajectory.data = tracker_output(tracker_output(:,2)==id,:);
        feet_world = image2world(feetPosition(trajectory.data(:,[3:6])), iCam);
        trajectory.data(:,[7,8]) = feet_world;
        trajectory.mcid = count;
        trajectory.scid = id;
        trajectory.camera = iCam;
        trajectory.startFrame = min(trajectory.data(:,1));
        trajectory.endFrame = max(trajectory.data(:,1));
        trajectories(count).trajectories = trajectory;
        count = count + 1;
        
    end
end

