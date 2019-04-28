clc
clear

opts = get_opts_aic();
% scene = 4;
% for iCam = opts.cams_in_scene{scene}
%     image = opts.reader.getFrame(iCam,1);
%     imwrite(rgb2gray(image),fullfile(sprintf('%s/aic_label_det/background', opts.experiment_root),sprintf('c%02d.jpg',iCam)))
% end

for iCam = 5
    imageROI        = imread(fullfile(sprintf('%s/aic_label_det/background', opts.experiment_root),sprintf('c%02d.jpg',iCam)));
    imageROI        = ~imbinarize(imageROI,1-10^-12);
    imwrite(imageROI,sprintf('%s/ROIs/background/c%02d.jpg', opts.dataset_path,  iCam));
end