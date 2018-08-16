% Demo visualizing OpenPose detections
opts = get_opts();

frame = 245000;
cam = 2;

load(fullfile(opts.dataset_path, 'detections', 'OpenPose', sprintf('camera%d.mat',cam)));

%% 

img = opts.reader.getFrame(cam, frame);
poses = detections(detections(:,1) == cam & detections(:,2) == frame,3:end);

img = renderPoses(img, poses);
% Transform poses into boxes
bboxes = [];
for i = 1:size(poses,1)
    bboxes(i,:) = pose2bb(poses(i,:), opts.render_threshold);
    bboxes(i,:) = scale_bb(bboxes(i,:), poses(i,:), 1.25);
end
img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
figure, imshow(img);



