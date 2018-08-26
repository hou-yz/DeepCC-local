function gen_det_dataset(opts,iCam)
detection_type='OpenPose';

sequence_window   = opts.sequence_intervals{opts.sequence};
start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));

% Load OpenPose detections for current camera
load(fullfile(opts.dataset_path, 'detections',detection_type, sprintf('camera%d.mat',iCam)));
in_time_range_ids = detections(:,2)>=start_frame & detections(:,2)<=end_frame;
detections   = detections(in_time_range_ids,:);

poses = detections;
% Convert poses to detections
detections = zeros(size(poses,1),6);
for k = 1:size(poses,1)
    pose = poses(k,3:end);
    bb = pose2bb(pose, opts.render_threshold);
    [newbb, newpose] = scale_bb(bb,pose,1.25);
    detections(k,:) = [iCam, poses(k,2), newbb];
end

video_name = fullfile(opts.dataset_path, 'videos', sprintf('camera%d.mp4', iCam));
videoObject = VideoReader(video_name);
videoObject.CurrentTime = (start_frame-1) / videoObject.FrameRate;

if exist(fullfile(opts.dataset_path, sprintf('det_dataset_%s',detection_type)),'dir') == 0
    mkdir(fullfile(opts.dataset_path, sprintf('det_dataset_%s',detection_type)));
end

for frame = start_frame : end_frame 
    t_init=tic;
    
    image = readFrame(videoObject);
    t_readframe=toc(t_init);
    
    det_ids = find(detections(:,2) == frame);
    if isempty(det_ids)
        continue;
    end
    det_in_frame = (detections(det_ids,:));
    bboxes = det_in_frame(:,3:end);
    for i = 1:size(det_in_frame,1)
        if bboxes(i,3) < 20 || bboxes(i,4) < 20
            det_image=uint8(zeros(256,128,3));
        else
            det_image=get_bb(image, bboxes(i,:));
            det_image=imresize(image,[256,128]);
        end
        imwrite(det_image,strcat(opts.dataset_path,sprintf('det_dataset_%s/c%d_f%06d_%04d.jpg',detection_type,iCam,frame,i)))
    end
    
    t_write=toc(t_init);
%     identities  = gt_in_frame(:, 2);
%     positions   = [gt_in_frame(:, 4),  gt_in_frame(:, 5), gt_in_frame(:, 6), gt_in_frame(:, 7)];
%     pic = insertObjectAnnotation(image,'rectangle', ...
%             positions, identities,'TextBoxOpacity', 0.8, 'FontSize', 13, 'Color', 255*[1,0,0] );
%     imshow(pic); 
%     drawnow
end



end
