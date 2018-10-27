clc
clear
opts = get_opts();
load('D:\MATLAB\Data\DukeMTMC\ground_truth\trainval.mat')
% [camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY]
for iCam = 1:8
    line_id = trainData(:,1)==iCam;
    trainData(line_id==1,3) = local2global(opts.start_frames(iCam), trainData(line_id==1,3));
end

% 150s
same_id_same_cam_threshold = 150;

ids = unique(trainData(:,2));
same_track_cam = zeros(length(ids),8);
same_track_cam_matrix = zeros(8,8);
same_track_intervals = cellmat(8,8,0,0,0);
consecutive_cam = zeros(length(ids),8);
consecutive_cam_matrix = zeros(8,8);  % LINES (i): out from iCam (i) into iCam (j): COLUMN (j)
same_id_same_cam_outage_times=[];
iCam_i_reintro_time_lists = cellmat(8,8,0,0,0);

num_optimal_path = 0;
num_non_optimal_path = 0;
length_optimal_path = 0;
length_non_optimal_path = 0;


for i = 1:length(ids)
    id = ids(i);
    lines = trainData(trainData(:,2)==id,:);
    lines = sortrows(lines,3);
    switch_cam_lines_id = [1;(lines(1:end-1,1)==lines(2:end,1))==0];
    % also add person_id who leave and enter the same camera after a while 
    switch_cam_lines_id(([0;(lines(1:end-1,3)+same_id_same_cam_threshold<lines(2:end,3)).*(lines(1:end-1,1)==lines(2:end,1))])==1)=1;
%     switch_cam_lines_id(logical([1;lines(1:end-1,3)==lines(2:end,3)]))=0;
%     switch_cam_lines_id(logical([lines(1:end-1,3)==lines(2:end,3);1]))=0;

    all=1:length(switch_cam_lines_id);
    intro_lines_id = all(switch_cam_lines_id==1);
    outro_lines_id = all([intro_lines_id(2:end)-1,end]);
    
    start_point = lines(intro_lines_id,[8,9]);
    end_point = lines(outro_lines_id,[8,9]);
    
    duration = lines(outro_lines_id,3) - lines(intro_lines_id,3);
    
    L1 = sqrt(sum((end_point(1:end-1,:) - start_point(2:end,:)).^2,2));
    L2 = sqrt(sum((end_point(1:end-1,:) - end_point(2:end,:)).^2,2));
    L3 = sqrt(sum((start_point(1:end-1,:) - start_point(2:end,:)).^2,2));
    
    tmp = (L1 < L2) .* (L1 < L3);
    correct_duration = sum(duration(logical([1;tmp])));
    false_duration = sum(duration)-correct_duration;
    
    num_optimal_path = num_optimal_path + sum(tmp);
    num_non_optimal_path = num_non_optimal_path + length(L1)-sum(tmp);
    
    length_optimal_path = length_optimal_path+correct_duration;
    length_non_optimal_path = length_non_optimal_path+false_duration;
    
    cams = lines(intro_lines_id,1);
    in_times  = repmat(lines(intro_lines_id,3),1,length(intro_lines_id))';
    out_times = repmat(lines(outro_lines_id,3),1,length(outro_lines_id));
    intervals = max(in_times-out_times,0);
    intervals = intervals+intervals';
    
    
    % reintro_logging
    intro_lines_id = intro_lines_id(2:end);
    outro_lines_id = intro_lines_id-1;
    
    cams_reintro_time = lines(intro_lines_id,3)-lines(outro_lines_id,3);
    if length(cams)~=length(unique(cams))
        all=1:length(switch_cam_lines_id);
        % see if due to within cam outage time
        switch_line_id=all(([0;(lines(1:end-1,3)+same_id_same_cam_threshold<lines(2:end,3)).*(lines(1:end-1,1)==lines(2:end,1))])==1);
        if isempty(switch_line_id)
            continue
        end
        % print image
%         figure(1)
%         img = opts.reader.getFrame(lines(switch_line_id(end)-1,1), global2local(opts.start_frames(lines(switch_line_id(end)-1,1)), lines(switch_line_id(end)-1,3)));
%         imshow(img(lines(switch_line_id(end)-1,5):lines(switch_line_id(end)-1,5)+lines(switch_line_id(end)-1,7),lines(switch_line_id(end)-1,4):lines(switch_line_id(end)-1,4)+lines(switch_line_id(end)-1,6),:))
%         figure(2)
%         img = opts.reader.getFrame(lines(switch_line_id(end),1), global2local(opts.start_frames(lines(switch_line_id(end),1)), lines(switch_line_id(end),3)));
%         imshow(img(lines(switch_line_id(end),5):lines(switch_line_id(end),5)+lines(switch_line_id(end),7),lines(switch_line_id(end),4):lines(switch_line_id(end),4)+lines(switch_line_id(end),6),:))
        
        same_id_same_cam_outage_time = lines(switch_line_id(end),3) - lines(switch_line_id(end)-1,3);
        same_id_same_cam_outage_times = [same_id_same_cam_outage_times;lines(switch_line_id(end),1:3),same_id_same_cam_outage_time];
    end
    consecutive_cam(i,1:length(cams))=cams;
    for j = 1:length(cams)-1
        consecutive_cam_matrix(cams(j),cams(j+1)) = consecutive_cam_matrix(cams(j),cams(j+1)) + 1;
        iCam_i_reintro_time_lists{cams(j),cams(j+1)} = [iCam_i_reintro_time_lists{cams(j),cams(j+1)},cams_reintro_time(j)];
    end
    same_track_cam(i,cams) = 1;
%     same_track_cam_matrix(cams,cams) = same_track_cam_matrix(cams,cams) + 1;
    for j = 1:length(cams)
        cam1=cams(j);
        for k = 1:length(cams)
            cam2=cams(k);
            same_track_intervals{cam1,cam2} = [same_track_intervals{cam1,cam2},intervals(j,k)];
           same_track_cam_matrix(cam1,cam2) = same_track_cam_matrix(cam1,cam2)+1;
        end
    end
    
        
end
