function gen_det_function(opts,iCam)
detection_type='OpenPose';

sequence_window   = global2local(opts.start_frames(iCam),opts.sequence_intervals{opts.sequence});

% Load OpenPose detections for current camera
load(fullfile(opts.dataset_path, 'detections',detection_type, sprintf('camera%d.mat',iCam)));
in_time_range_ids = ismember(detections(:,2),sequence_window);
detections   = detections(in_time_range_ids,:);

poses = detections;
% Convert poses to detections
detections = zeros(size(poses,1),6);
for k = 1:size(poses,1)
    pose = poses(k,3:end);
    bb = pose2bb(pose, opts.render_threshold);
    [newbb, ~] = scale_bb(bb,pose,1.25);
    detections(k,:) = [iCam, poses(k,2), newbb];
end

folder_dir = fullfile(opts.dataset_path, 'ALL_det_bbox', sprintf('det_bbox_%s_%s',detection_type,opts.sequence_names{opts.sequence}));
if exist(folder_dir,'dir') == 0
    mkdir(folder_dir);
end

for i = 1:length(sequence_window)
    frame = sequence_window(i);
    image = opts.reader.getFrame(iCam,frame);

    det_ids = find(detections(:,2) == frame);
    if isempty(det_ids)
        continue;
    end
    det_in_frame = (detections(det_ids,:));
    bboxes = det_in_frame(:,3:end);
    for i = 1:size(det_in_frame,1)
        if bboxes(i,3) < 20 || bboxes(i,4) < 20
            det_image=uint8(zeros(1,1,3));
        else
            det_image=get_bb(image, bboxes(i,:));
        end
        imwrite(det_image,fullfile(folder_dir,sprintf('c%d_f%06d_%04d.jpg',iCam,frame,i)))
    end
end

end
