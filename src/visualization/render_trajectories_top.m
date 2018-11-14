clc
clear
opts = get_opts();

% Load ground truth trajectories (or your own)
load(fullfile(opts.dataset_path, 'ground_truth', 'trainval.mat'));
trajectories = trainData;
trajectories = trajectories(trajectories(:,2)==725,:);

% Load map
map = imread('src/visualization/data/map.jpg');

% Create folder
folder = 'video-results';
mkdir([folder]);
video_name = fullfile(opts.experiment_root, folder, 'duke_topview.mp4');

% Params
% colors     = distinguishable_colors(1000);

colors = [38,35,226;251,128,114;253,180,98;152,78,163];%206,18,86
tail_size  = inf;
fps        = 120;
rois       = opts.ROIs;
ids        = unique(trajectories(:,2));
interval   = opts.sequence_intervals{1};
startFrame = interval(1);
endFrame   = interval(end);

% Convert frames to global clock
for iCam = 1:8
    inds = find(trajectories(:,1) == iCam);
    trajectories(inds,3) = local2global(opts.start_frames(iCam),trajectories(inds,3));
end

trajectories = sortrows(trajectories,3);
% 150 frames
same_id_same_cam_threshold = 150;
switch_cam_lines_id = [1;(trajectories(1:end-1,1)==trajectories(2:end,1))==0];
% also add person_id who leave and enter the same camera after a while 
switch_cam_lines_id(([0;(trajectories(1:end-1,3)+same_id_same_cam_threshold<trajectories(2:end,3)).*(trajectories(1:end-1,1)==trajectories(2:end,1))])==1)=1;

all=1:length(switch_cam_lines_id);
intro_lines_id = all(switch_cam_lines_id==1);
outro_lines_id = all([intro_lines_id(2:end)-1,end]);

for i =1:length(intro_lines_id)
    trajectories(intro_lines_id(i):outro_lines_id(i),2) = i;
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

figure()
%% Top View
for frame = endFrame:endFrame
    fprintf('Frame %d/%d\n', frame, endFrame);
    
    img = map;
    data = trajectories;
    ids = unique(data(:,2));
    
    polylines = [];
    polycolors = [];
    
    for k = 1:length(ids)
        id = ids(k);
        mask = logical((trajectories(:,2) == id));% .* (trajectories(:,3) >= frame - tail_size) .* (trajectories(:,3) < frame)
        
        mapped = world2map(trajectories(mask, [8 9]));
        polylines{k} = mapped;
        polycolors{k} = colors(id,:);
%         polycolors{k} = colors(1+mod(id,1000),:);
        img = cv.polylines(img, polylines{k}, 'Closed', false, 'Thickness', 4, 'Color', polycolors{k});
    end
    img = rot90(img,3);
%     for k = 1:length(ids)
%         pos = [0,size(img,2)];
%         pos = pos+[1,-1].*polylines{k}(end,:);
%         img = insertText(img,pos([2,1]), sprintf('%d',k),'FontSize',20, 'BoxColor','black','TextColor', 255*polycolors{k});
%     end 
end
imwrite(img,'non-optimal_path.png')
imshow(img)