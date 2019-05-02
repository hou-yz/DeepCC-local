clc
clear

opts = get_opts_aic();
all_colors = distinguishable_colors(1000)*255;

%% print background
for scene = 1:5
mkdir(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene));
for iCam = opts.cams_in_scene{scene}
    %% load image background
    imageROI = opts.reader.getFrame(scene,iCam,1);
    
    %% load gt
    if scene~=2 && scene~=5
    gt = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));
    circles = gt(:,3:4)+0.5*gt(:,5:6);
    circles(:,3) = 3;
    colors = all_colors(gt(:,2),:);
    imageROI = insertShape(imageROI,'FilledCircle',circles,'Color', colors);
    end
    imwrite(imageROI,  fullfile(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene),sprintf('c%02d.png',iCam)))
end
end


%% color to binary
% for scene = 1:5
% mkdir(sprintf('%s/ROIs/background/S%02d', opts.dataset_path, scene));
% for iCam = opts.cams_in_scene{scene}
%     imageROI        = imread(fullfile(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene),sprintf('c%02d.png',iCam)));
%     imageROI        = rgb2gray(imageROI);
%     imageROI        = ~imbinarize(imageROI,1-10^-12);
%     imwrite(imageROI,sprintf('%s/ROIs/background/S%02d/c%02d.png', opts.dataset_path, scene, iCam));
% end
% end