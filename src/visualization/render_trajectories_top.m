opts = get_opts();

% Load ground truth trajectories (or your own)
load(fullfile(opts.dataset_path, 'ground_truth', 'trainval.mat'));
trajectories = trainData;

% Load map
map = imread('src/visualization/data/map.jpg');

% Create folder
folder = 'video-results';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);
video_name = fullfile(opts.experiment_root, opts.experiment_name, folder, 'duke_topview.mp4');

% Params
colors     = distinguishable_colors(1000);
tail_size  = 300;
fps        = 120;
rois       = opts.ROIs;
ids        = unique(trajectories(:,2));
interval   = opts.sequence_intervals{opts.sequence};
startFrame = interval(1);
endFrame   = interval(end);

% Convert frames to global clock
for iCam = 1:8
    inds = find(trajectories(:,1) == iCam);
    trajectories(inds,3) = local2global(opts.start_frames(iCam),trajectories(inds,3));
end

% Delineate the regions of interest
roimask = ones(size(map,1),size(map,2));
for k = 1:8
    roi = rois{k};
    mapped = world2map(image2world(roi,k));
    map = cv.polylines(map, mapped, 'Thickness',1.5);
    roimask = cv.fillPoly(roimask, mapped);
end
roimask = 1-roimask/13;
map = double(map);
map(:,:,1) = map(:,:,1) .* roimask;
map(:,:,2) = map(:,:,2) .* roimask;
map(:,:,3) = map(:,:,3) .* roimask;
map = uint8(map);

%% Top View

vid = VideoWriter(video_name, 'MPEG-4');
open(vid);
for frame = startFrame:fps:endFrame
    fprintf('Frame %d/%d\n', frame, endFrame);
    
    img = map;
    data = trajectories(trajectories(:,3) == frame,:);
    ids = unique(data(:,2));
    
    polylines = [];
    polycolors = [];
    
    for k = 1:length(ids)
        id = ids(k);
        mask = logical((trajectories(:,2) == id) .* (trajectories(:,3) >= frame - tail_size) .* (trajectories(:,3) < frame));
        
        mapped = world2map(trajectories(mask, [8 9]));
        polylines{k} = mapped;
        polycolors{k} = colors(1+mod(id,1000),:);
        img = cv.polylines(img, polylines{k}, 'Closed', false, 'Thickness', 4, 'Color', 255*colors(1+mod(id,1000),:));
        
    end
    img = rot90(img,3);
    img = insertText(img,[0 0], sprintf('Frame %d', frame),'FontSize',20);
    writeVideo(vid,img);
    
end
close(vid);
