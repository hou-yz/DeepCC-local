clc
clear

opts = get_opts_aic();
opts.experiment_name = 'aic_label_det';
opts.sequence = 6;
scene = 5;
cams = 34%[27,28,33,34,35];
for i = 1:length(cams)%1:length(opts.cams_in_scene{scene})
    iCam = cams(i)%opts.cams_in_scene{scene}(i);
    resdata = dlmread(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.folder_by_seq{opts.sequence}));
    %% det bboxs to og size
%     bboxs = resdata(:,3:6);
%     og_wh = bboxs(:,3:4)/1.4;
%     bboxs(:,1:2) = bboxs(:,1:2)+0.2*og_wh;
%     bboxs(:,3:4) = og_wh;
%     %% det bboxs to gt size
%     bboxs(:,1:2) = bboxs(:,1:2) - 20;
%     bboxs(:,3:4) = bboxs(:,3:4) + 40;
%     resdata(:,3:6) = bboxs;
    %% unique bbox for each id in each frame
    unique_ids = unique(resdata(:,2));
    to_remove = [];
    for i = 1:length(unique_ids)
        id = unique_ids(i);
        line_idx = find(resdata(:,2)==id);
        detections = resdata(line_idx,:);
        frames = detections(:,1);
        [unique_frames, I] = unique(frames, 'first');
        dup_idx = 1:length(frames);
        dup_idx(I) = [];
        duplicate_frames = frames(dup_idx);
        if isempty(duplicate_frames)
            continue
        end
        for j = 1:length(duplicate_frames)
            frame = duplicate_frames(j);
            dup_line_ids = find(frames==frame);
            resdata(line_idx(dup_line_ids(1)),:) = mean(detections(dup_line_ids,:),1);
        end
        to_remove = [to_remove;line_idx(dup_idx)];
    end
    resdata(to_remove,:) = [];
    dlmwrite(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.folder_by_seq{opts.sequence}),...
        resdata, 'delimiter', ' ', 'precision', 6);
end