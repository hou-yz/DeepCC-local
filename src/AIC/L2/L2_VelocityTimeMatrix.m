function [velocityChangeMatrix,timeIntervalMatrix,iousMatrix,consider_matrix] = L2_VelocityTimeMatrix(trackletData,intervalLength)

velocityChangeMatrix = zeros(length(trackletData));
timeIntervalMatrix = zeros(length(trackletData));
iousMatrix = zeros(length(trackletData));

startFrame = zeros(1,length(trackletData));
endFrame = zeros(1,length(trackletData));
% headData = struct('data',[]);
% tailData = struct('data',[]);
structData = struct('data',[]);
start_bbox = zeros(length(trackletData),4);
end_bbox = zeros(length(trackletData),4);
for i = 1:length(trackletData)
frames = trackletData{i}(:,1);
startFrame(i) = min(frames);
endFrame(i) = max(frames);
% headData(i).data = trackletData{i}(frames<frames(1)+intervalLength,:);
% tailData(i).data = trackletData{i}(frames>frames(end)-intervalLength,:);
structData(i).data = trackletData{i};
start_bbox(i,:) = max(trackletData{i}(1,3:6),1);
end_bbox(i,:) = max(trackletData{i}(end,3:6),1);
end

start_le_end = startFrame>=endFrame';
% start_le_end = logical(triu(ones(length(trackletData)),1));
%% time
%tail_start_frame - head_end_frame
time_interval = startFrame - endFrame'-1;
timeIntervalMatrix(start_le_end) = time_interval(start_le_end);
timeIntervalMatrix = (timeIntervalMatrix + timeIntervalMatrix');
%% velocity
[~, ~, tail_startpoint, head_endpoint, ~, ~, Velocity] = getTrackletFeatures(structData);
% velocity_diff_x = headVelocity(:,1) - tailVelocity(:,1)';
% velocity_diff_y = headVelocity(:,2) - tailVelocity(:,2)';
% velocity_diff = sqrt(velocity_diff_x.^2 + velocity_diff_y.^2);
% velocityChangeMatrix(start_le_end) = velocity_diff(start_le_end);
% velocityChangeMatrix = velocityChangeMatrix + velocityChangeMatrix';
distance_x = tail_startpoint(:,1) - head_endpoint(:,1)';
distance_y = tail_startpoint(:,2) - head_endpoint(:,2)';
diff_x = min(distance_x./Velocity(:,1),distance_x./(Velocity(:,1)'));
diff_y = min(distance_y./Velocity(:,2),distance_y./(Velocity(:,2)'));
diff = sqrt(diff_x.^2 + diff_y.^2)./(1+timeIntervalMatrix);
velocityChangeMatrix(start_le_end) = diff(start_le_end);
velocityChangeMatrix = velocityChangeMatrix + velocityChangeMatrix';

%% iou
ious = bboxOverlapRatio(start_bbox,end_bbox);
iousMatrix(start_le_end) = ious(start_le_end);
iousMatrix = (iousMatrix + iousMatrix');
% iousMatrix(iousMatrix==0)=-0.5;

%%
consider_matrix = start_le_end + start_le_end';
end

