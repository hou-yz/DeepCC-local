clc
clear
opts = get_opts();
load('D:\MATLAB\Data\DukeMTMC\ground_truth\trainval.mat')
% [camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY]
for iCam = 1:8
    line_id = trainData(:,1)==iCam;
    trainData(line_id==1,3) = local2global(opts.start_frames(iCam), trainData(line_id==1,3));
end

% 30s
same_id_same_cam_threshold = 60*30;

ids = unique(trainData(:,2));
same_track_cam = zeros(length(ids),8);
same_track_cam_matrix = zeros(8,8);
consecutive_cam = zeros(length(ids),8);
consecutive_cam_matrix = zeros(8,8);
same_id_same_cam_outage_times=[];
for i = 1:length(ids)
    id = ids(i);
    lines = trainData(trainData(:,2)==id,:);
    lines = sortrows(lines,3);
    switch_cam_lines_id = [1;(lines(1:end-1,1)==lines(2:end,1))==0];
    % also add person_id who leave and enter the same camera after a while 
    switch_cam_lines_id(([0;(lines(1:end-1,3)+same_id_same_cam_threshold<lines(2:end,3)).*(lines(1:end-1,1)==lines(2:end,1))])==1)=1;
%     switch_cam_lines_id(logical([1;lines(1:end-1,3)==lines(2:end,3)]))=0;
%     switch_cam_lines_id(logical([lines(1:end-1,3)==lines(2:end,3);1]))=0;

    cams = lines(switch_cam_lines_id==1,1);
    if length(cams)~=length(unique(cams))
        all=1:length(switch_cam_lines_id);
        % see if due to within cam outage time
        switch_line=all(([0;(lines(1:end-1,3)+same_id_same_cam_threshold<lines(2:end,3)).*(lines(1:end-1,1)==lines(2:end,1))])==1);
        if isempty(switch_line)
            continue
        end
        % print image
        figure(1)
        img = opts.reader.getFrame(lines(switch_line(end)-1,1), global2local(opts.start_frames(lines(switch_line(end)-1,1)), lines(switch_line(end)-1,3)));
        imshow(img(lines(switch_line(end)-1,5):lines(switch_line(end)-1,5)+lines(switch_line(end)-1,7),lines(switch_line(end)-1,4):lines(switch_line(end)-1,4)+lines(switch_line(end)-1,6),:))
        figure(2)
        img = opts.reader.getFrame(lines(switch_line(end),1), global2local(opts.start_frames(lines(switch_line(end),1)), lines(switch_line(end),3)));
        imshow(img(lines(switch_line(end),5):lines(switch_line(end),5)+lines(switch_line(end),7),lines(switch_line(end),4):lines(switch_line(end),4)+lines(switch_line(end),6),:))
        
        same_id_same_cam_outage_time = lines(switch_line(end),3) - lines(switch_line(end)-1,3);
        same_id_same_cam_outage_times = [same_id_same_cam_outage_times;lines(switch_line(end),1:3),same_id_same_cam_outage_time];
    end
    consecutive_cam(i,1:length(cams))=cams;
    for j = 1:length(cams)-1
        consecutive_cam_matrix(cams(j),cams(j+1)) = consecutive_cam_matrix(cams(j),cams(j+1)) + 1;
    end
    same_track_cam(i,cams) = 1;
    same_track_cam_matrix(cams,cams) = same_track_cam_matrix(cams,cams) + 1;
end