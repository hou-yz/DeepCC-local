function [velocityChangeMatrix,distanceMatrix,shapeChangeMatrix,iousMatrix,timeIntervalMatrix, impossibilityMatrix] = aic_VelocityTimeMatrix(opts,trackletData,intervalLength)

velocityChangeMatrix = zeros(length(trackletData));
distanceMatrix = zeros(length(trackletData));
shapeChangeMatrix = zeros(length(trackletData));
timeIntervalMatrix = zeros(length(trackletData));
iousMatrix = zeros(length(trackletData));

startFrame = zeros(1,length(trackletData));
endFrame = zeros(1,length(trackletData));
headGPS = cell(1,length(trackletData));
tailGPS = cell(1,length(trackletData));
% structData = struct('data',[]);
start_bbox = zeros(length(trackletData),4);
end_bbox = zeros(length(trackletData),4);
start_bbox_size = zeros(1,length(trackletData));
end_bbox_size = zeros(1,length(trackletData));
bbox_hw = zeros(length(trackletData),2);
for i = 1:length(trackletData)
frames = trackletData{i}(:,1);
startFrame(i) = min(frames);
endFrame(i) = max(frames);
headGPS{i} = trackletData{i}(frames<frames(1)+intervalLength,7:8);
tailGPS{i} = trackletData{i}(frames>frames(end)-intervalLength,7:8);
% structData(i).data = trackletData{i};
start_bbox(i,:) = max(trackletData{i}(1,3:6),1);
end_bbox(i,:) = max(trackletData{i}(end,3:6),1);
start_bbox_size(i) = sqrt(trackletData{i}(1, 5).^ 2 + trackletData{i}(1, 6).^ 2);
end_bbox_size(i) = sqrt(trackletData{i}(end, 5).^ 2 + trackletData{i}(end, 6).^ 2);
bbox_hw(i,:) = mean(max(trackletData{i}(end,5:6),1));
end

% start_le_end = startFrame>=endFrame';
consecutive = startFrame==endFrame'+1;
start_le_end = logical(triu(ones(length(trackletData)),1));
%% time
%tail_start_frame - head_end_frame
time_interval = endFrame-startFrame';
timeIntervalMatrix(start_le_end) = time_interval(start_le_end);
timeIntervalMatrix = (timeIntervalMatrix + timeIntervalMatrix');

% [~, ~, tail_startpoint, head_endpoint, ~, ~, Velocity] = getTrackletFeatures(structData);
[head_startpoint, head_endpoint, headVelocity,headIntervals] = getGpsSpeed( trackletData);
[tail_startpoint, tail_endpoint, tailVelocity,tailIntervals] = getGpsSpeed( trackletData);
%% distance
distance_x = tail_endpoint(:,1) - head_startpoint(:,1)';
distance_y = tail_endpoint(:,2) - head_startpoint(:,2)';
diff_x = distance_x;
diff_y = distance_y;
% diff_x = distance_x-headVelocity(:,1) + distance_x-(tailVelocity(:,1)');
% diff_y = distance_y-headVelocity(:,2) + distance_y-(tailVelocity(:,2)');
distance = sqrt(diff_x.^2 + diff_y.^2);
% diff = 0.5*(diff./start_bbox_size + diff./end_bbox_size');
distanceMatrix(start_le_end) = distance(start_le_end);
distanceMatrix = distanceMatrix + distanceMatrix';
%% velocitychange
neededVelocity_x = diff_x./time_interval*10;
neededVelocity_y = diff_y./time_interval*10;
neededVelocity = sqrt(neededVelocity_x.^2 + neededVelocity_y.^2);
velocity_diff_x = abs(headVelocity(:,1)-tailVelocity(:,1)') + abs(neededVelocity_x-headVelocity(:,1))+abs(tailVelocity(:,1)'-neededVelocity_x);
velocity_diff_y = abs(headVelocity(:,2)-tailVelocity(:,2)') + abs(neededVelocity_y-headVelocity(:,2))+abs(tailVelocity(:,2)'-neededVelocity_y);
velocity_diff = sqrt(velocity_diff_x.^2 + velocity_diff_y.^2);
velocityChangeMatrix(start_le_end) = velocity_diff(start_le_end);
velocityChangeMatrix = velocityChangeMatrix + velocityChangeMatrix';
velocityChangeMatrix(isnan(velocityChangeMatrix)) = 0;
velocityChangeMatrix(isinf(velocityChangeMatrix)) = 0;
%% shape change
shapeChange = abs(start_bbox_size - end_bbox_size');
shapeChange = max(shapeChange./start_bbox_size,shapeChange./(end_bbox_size'));
shapeChangeMatrix(start_le_end) = shapeChange(start_le_end);
shapeChangeMatrix = shapeChangeMatrix + shapeChangeMatrix';
%% iou
ious = bboxOverlapRatio(start_bbox,end_bbox);
iousMatrix(consecutive) = ious(consecutive);
iousMatrix = iousMatrix + iousMatrix';
% iousMatrix(iousMatrix==0)=-0.5;
%% impossible
overlapping     = pdist2(headIntervals,tailIntervals, @overlapTest);
merging         = (distanceMatrix < 6) & overlapping & (ious > 0);
velocity        = distanceMatrix./(abs(timeIntervalMatrix)+10^-12)*10;
violators       = velocity > opts.trajectories.speed_limit;

% build impossibility matrix
impossibilityMatrix = zeros(length(trackletData));
impossibilityMatrix(violators & merging ~=1) = 1;
impossibilityMatrix(overlapping & merging ~=1) = 1;

%%
consider_matrix = start_le_end + start_le_end';
end


function overlap = overlapTest(interval1, interval2)

duration1       = interval1(2) - interval1(1);
duration2       = interval2(:,2) - interval2(:, 1);

i1              = repmat(interval1,size(interval2,1), 1);
unionMin        = min([i1, interval2], [], 2);
unionMax        = max([i1, interval2], [], 2);

overlap         = double(duration1 + duration2 - unionMax + unionMin >= 0);

end

