function trajectories = loadL2trajectories(opts, scene)
% Loads single camera trajectories that have been written to disk
trajectories = [];
count = 1;
if opts.dataset == 0
    cam_pool = 1:opts.num_cam;
elseif opts.dataset == 2
    cam_pool = opts.cams_in_scene{scene};
end
for iCam = cam_pool
    if opts.dataset <= 1
        tracker_output = dlmread(fullfile(opts.experiment_root, opts.experiment_name, 'L2-trajectories', sprintf('cam%d_%s.txt',iCam, opts.sequence_names{opts.sequence})));
    elseif opts.dataset == 2
        tracker_output = dlmread(fullfile(opts.experiment_root, opts.experiment_name, 'L2-trajectories', sprintf('cam%d_%s_no_wait.txt',iCam, opts.sequence_names{opts.sequence})));
    end
    ids = unique(tracker_output(:,2));
    
    traj_for_iCam = load(fullfile(opts.experiment_root, opts.experiment_name, 'L2-trajectories', sprintf('trajectories%d_%s.mat',iCam, opts.sequence_names{opts.sequence})));
    traj_for_iCam = traj_for_iCam.trajectories;
    
    for idx = 1:length(ids)
        id = ids(idx);
        trajectory = [];
        trajectory.data = tracker_output(tracker_output(:,2)==id,:);
        if opts.dataset~=2
        feet_world = image2world(feetPosition(trajectory.data(:,[3:6])), iCam);
        trajectory.data(:,[7,8]) = feet_world;
        end
        trajectory.mcid = count;
        trajectory.scid = id;
        trajectory.camera = iCam;
        trajectory.startFrame = min(trajectory.data(:,1));
        trajectory.endFrame = max(trajectory.data(:,1));
        trajectory.feature = traj_for_iCam(id).feature;
        if opts.visualize
        i=floor(length(trajectory.data(:,1))/2);
        frame = round(trajectory.data(i,1));
        left = round(max(trajectory.data(i,3),1));
        top = round(max(trajectory.data(i,4),1));
        width = round(trajectory.data(i,5));
        height = round(trajectory.data(i,6));
        if opts.dataset ~= 2
            img = opts.reader.getFrame(iCam,frame);
        else
            img = opts.reader.getFrame(scene, iCam,frame);
        end
        img_size = size(img);
        trajectory.snapshot = img(top:min(top+height,img_size(1)),left:min(left+width,img_size(2)),:);
        end
        trajectories(count).trajectories = trajectory;
        count = count + 1;
        
    end
end

