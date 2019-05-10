clc
clear

opts = get_opts_aic();
opts.experiment_name = 'aic_zju_ensemble';
opts.sequence = 6;
all_colors = distinguishable_colors(5000)*255;

%% print background
% for scene = 5%1:5
% mkdir(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene));
% for iCam = 29%opts.cams_in_scene{scene}
%     %% load image background
%     imageROI = opts.reader.getFrame(scene,iCam,1);
%     
%     %% load gt
%     if scene~=2 && scene~=5
%         gt = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));
%     else
%         gt = load(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.folder_by_seq{opts.sequence}));
%     bboxs = gt(:,3:6);
%     % gt2det bbox
%     bboxs(:,1:2) = bboxs(:,1:2)+20; bboxs(:,3:4) = bboxs(:,3:4)-40; 
%     feet_pos = feetPosition(bboxs);
%     feet_pos(:,3) = 3;
%     colors = all_colors(gt(:,2),:);
%     imageROI = insertShape(imageROI,'FilledCircle',feet_pos,'Color', colors);
%     end
%     imwrite(imageROI,  fullfile(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene),sprintf('c%02d.png',iCam)))
% end
% end


%% color to binary
for scene = 5
mkdir(sprintf('%s/ROIs/background/S%02d', opts.dataset_path, scene));
for iCam = opts.cams_in_scene{scene}
    imageROI        = imread(fullfile(sprintf('%s/aic_label_det/background/S%02d', opts.experiment_root, scene),sprintf('c%02d.png',iCam)));
    imageROI        = rgb2gray(imageROI);
    imageROI        = ~imbinarize(imageROI,1-10^-12);
    imwrite(imageROI,sprintf('%s/ROIs/background/S%02d/c%02d.png', opts.dataset_path, scene, iCam));
end
end