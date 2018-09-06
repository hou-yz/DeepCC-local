function gen_gt_function(opts,iCam,fps)

sequence_window   = opts.sequence_intervals{opts.sequence};
start_frame       = global2local(opts.start_frames(iCam), sequence_window(1));
end_frame         = global2local(opts.start_frames(iCam), sequence_window(end));

ground_truth = load(fullfile(opts.dataset_path,'ground_truth/trainval.mat'));
ground_truth = ground_truth.trainData;
ground_truth = ground_truth(ground_truth(:,1)==iCam & ground_truth(:,3)>= start_frame & ground_truth(:,3)<= end_frame,:);

video_name = fullfile(opts.dataset_path, 'videos', sprintf('camera%d.mp4', iCam));
videoObject = VideoReader(video_name);
videoObject.CurrentTime = (start_frame-1) / videoObject.FrameRate;

folder_dir = fullfile(opts.dataset_path, sprintf('gt_bbox_%d_fps',fps));
if exist(folder_dir,'dir') == 0
    mkdir(folder_dir);
end
if exist(fullfile(folder_dir,sprintf('camera%d',iCam)),'dir') == 0
    mkdir(fullfile(folder_dir,sprintf('camera%d',iCam)));
end

frame = start_frame;
while frame < end_frame 
    for i = 1:60/fps
        image = readFrame(videoObject);
        frame = frame+1;
    end
    gt_ids = find(ground_truth(:,3) == frame);
    if isempty(gt_ids)
        continue;
    end
    gt_in_frame = (ground_truth(gt_ids,:));
    left = max(1,gt_in_frame(:,4));
    top = max(1,gt_in_frame(:,5));
    width = min(1920,gt_in_frame(:,6));
    height = min(1080,gt_in_frame(:,7));
    for i = 1:size(gt_in_frame,1)
        gt_image = image(top(i):top(i)+height(i),left(i):left(i)+width(i),:);
        
        imwrite(gt_image,fullfile(folder_dir,sprintf('camera%d/%04d_c%d_f%06d.jpg',iCam,gt_in_frame(i,2),iCam,frame)))
    end
end



end
