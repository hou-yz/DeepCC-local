opts = get_opts_aic;

opts.feature_dir = 'det_features_hsv_hist';

for i = 1:length(opts.seqs)

    iCam = opts.seqs(i);
    opts.current_camera = iCam;
%     detections          = load(sprintf('%s/train/S%02d/c%03d/det/det_%s.txt', opts.dataset_path, opts.current_scene, iCam, opts.detections));
    detections          = load(sprintf('%s/train/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.current_scene, iCam));
    
    features = zeros(length(detections), 36+2);
    features(:,1) = iCam;
    features(:,2) = detections(:,2);
    
    frameCurrent = -1;
    for j = 1:length(detections)
        if detections(j, 1) ~= frameCurrent
            counter = 0;
            frameCurrent = detections(j, 1);
        else
            counter = counter + 1;
        end
%         fig = imread(sprintf('%s/ALL_det_bbox/val/s%02d_c%02d_f%05d_%03d.jpg', opts.dataset_path, opts.current_scene, iCam, detections(j, 1), counter));
        fig = imread(sprintf('%s/ALL_gt_bbox/val/gt_bbox_10_fps/%04d_s%02d_c%02d_f%05d.jpg', opts.dataset_path, detections(j, 2), opts.current_scene, iCam, detections(j, 1)));
        features(j,3:end) = extractHSVHistogram(fig);
        
    end
    features = features';
    
    h5create(fullfile(opts.dataset_path, 'L0-features', opts.feature_dir, sprintf('features%d.h5',iCam)), '/emb', [38 length(features)]);
    h5write(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb',features);
    disp(sprintf('features%d is completed', iCam));
    
end