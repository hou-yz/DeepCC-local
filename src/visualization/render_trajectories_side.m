opts = get_opts();

% Load ground truth trajectories (or your own)
load(fullfile(opts.dataset_path, 'ground_truth', 'trainval.mat'));
trajectories = trainData;

% Create folder
folder = 'video-results';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);
video_name = fullfile(opts.experiment_root, opts.experiment_name, folder, 'duke_sideview.mp4');


% Params
colors    = distinguishable_colors(1000);
tail_size = 300;
fps       = 120;
rois      = opts.ROIs;
ids       = unique(trajectories(:,2));

% Convert frames to global clock
for iCam = 1:8
    inds = find(trajectories(:,1) == iCam);
    trajectories(inds,3) = local2global(opts.start_frames(iCam),trajectories(inds,3));
end


%% Render side view

vid = VideoWriter(video_name, 'MPEG-4');
open(vid);

% Placeholders for tags
inds = [271,1; 271, 481; 271, 961; 271,1441;  1,961; 1,1441; 1,481; 1,1];


for frame = startFrame:fps:endFrame
    fprintf('Frame %d/%d\n', frame, endFrame);
    
    data = trajectories(trajectories(:,3) == frame,:);
    ids = unique(data(:,2));
    finalimg = zeros(540,1920,3);
    
    % Will read through each camera separately (slow)
    % TODO: Render each camera separately, then combine into mosaic (fast)
    for iCam = 1:opts.num_cam
        
        img = opts.reader.getFrame(iCam, global2local(opts.start_frames(iCam),frame));
        
        % Shade ROI with blue
        roi = rois{iCam};
        roimask = ones(size(img,1),size(img,2));
        roimask = cv.fillPoly(roimask, roi);      
        img = double(img);
        c3 = img(:,:,3);
        c3(logical(~roimask)) = 128 + 0.5*c3(logical(~roimask));
        img(:,:,3) = c3;
        img = uint8(img);
        
        % Draw all tails for current camera frame
        for k = 1:length(ids)
            id = ids(k);
            mask = logical((trajectories(:,1) == iCam) .* (trajectories(:,2) == id) .* (trajectories(:,3) >= frame - tail_size) .* (trajectories(:,3) < frame));
            bb= trajectories(mask, [4 5 6 7]);
            feet = feetPosition(bb);
            if ~isempty(bb)
                img = insertObjectAnnotation(img,'rectangle', bb(end,:), id, 'Color', 255*colors(1+mod(id,1000),:));
            end
            
            img = cv.polylines(img, feet, 'Closed', false, 'Thickness', 4, 'Color', 255*colors(1+mod(id,1000),:));
            
        end
        img = imresize(img,0.25);
        % Add image to mosaic
        finalimg(inds(iCam,1):inds(iCam,1)+270-1,inds(iCam,2):inds(iCam,2)+480-1,:) = img;
    end
    
    % Add camera tags
    finalimg = uint8(finalimg);
    finalimg = insertText(finalimg,[0 510], sprintf('Frame %d', frame),'FontSize',20);
    finalimg = insertText(finalimg,[0 0], 'Camera 8', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[480 0], 'Camera 7', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[960 0], 'Camera 5', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[1440 0], 'Camera 6', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[0 270], 'Camera 1', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[480 270], 'Camera 2', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[960 270], 'Camera 3', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    finalimg = insertText(finalimg,[1440 270], 'Camera 4', 'FontSize',20, 'BoxColor', 'black','BoxOpacity',0.8,'TextColor','white');
    
    % Write mosaic frame to video
    writeVideo(vid,finalimg);   
end
close(vid);