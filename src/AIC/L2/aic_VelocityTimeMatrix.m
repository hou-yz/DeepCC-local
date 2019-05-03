function [spacetimeAffinity, impossibilityMatrix] = aic_VelocityTimeMatrix(opts,trackletData, iCam,intervalLength)

spacetimeAffinity = zeros(length(trackletData));
distanceMatrix = zeros(length(trackletData));
shapeChangeMatrix = zeros(length(trackletData));
timeIntervalMatrix = zeros(length(trackletData));
iousMatrix = zeros(length(trackletData));

startFrame = zeros(length(trackletData),1);
endFrame = zeros(length(trackletData),1);
headGPS = cell(length(trackletData),1);
tailGPS = cell(length(trackletData),1);
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

consecutive = startFrame'==endFrame+1;
%% time
time_interval = endFrame'-startFrame;
consider = time_interval>0; consider_nt = ~consider'; % not & tracpose
timeIntervalMatrix(consider) = time_interval(consider);
% timeIntervalMatrix = (timeIntervalMatrix + timeIntervalMatrix');

% [~, ~, tail_startpoint, head_endpoint, ~, ~, Velocity] = getTrackletFeatures(structData);
[startpoint, endpoint, Velocity,Intervals] = getGpsSpeed( trackletData);
%% distance
A = true(length(trackletData));
distance_x = endpoint(:,1)' - startpoint(:,1);
distance_y = endpoint(:,2)' - startpoint(:,2);
needed_v_x = distance_x./(time_interval+10^-12)*10;
needed_v_y = distance_y./(time_interval+10^-12)*10;
needed_v_x(~consider) = 0; needed_v_y(~consider) = 0;
% needed_v_x(tril(A,-1)) = needed_v_x(triu(A,1))'; needed_v_y(tril(A,-1)) = needed_v_y(triu(A,1))';
distance = sqrt(distance_x.^2 + distance_y.^2);
distanceMatrix(consider) = distance(consider);
% distanceMatrix = distanceMatrix + distanceMatrix';
%% velocitychange
v_dist_euc = pdist2(Velocity,Velocity,'euclidean');
v_dist_cos = pdist2(Velocity,Velocity,'cosine');
needed_v_dist_euc = zeros(length(trackletData));
needed_v_dist_cos = zeros(length(trackletData));
for i = 1:length(trackletData)
    needed_v_i = [needed_v_x(i,:);needed_v_y(i,:)]';
    needed_v_dist_euc(i,:) = pdist2(Velocity(i,:),needed_v_i,'euclidean');
    needed_v_dist_cos(i,:) = pdist2(Velocity(i,:),needed_v_i,'cosine');    
end
%% shape change
shapeChange = abs(start_bbox_size - end_bbox_size');
shapeChange = max(shapeChange./start_bbox_size,shapeChange./(end_bbox_size'));
shapeChangeMatrix(consider) = shapeChange(consider);
% shapeChangeMatrix = shapeChangeMatrix + shapeChangeMatrix';
%% iou
ious = bboxOverlapRatio(start_bbox,end_bbox);
iousMatrix(consecutive) = ious(consecutive);
% iousMatrix = iousMatrix + iousMatrix';
% iousMatrix(iousMatrix==0)=-0.5;
%% impossible
overlapping     = pdist2(Intervals,Intervals, @overlapTest);
merging         = overlapping & (ious > 0);
violators       = zeros(length(trackletData));
% violators(v_dist_euc > opts.trajectories.speed_limit & distance > 5) = 1;
% violators(needed_v_dist_euc > opts.trajectories.speed_limit & distance > 5) = 1;
if ~ismember(iCam,opts.trajectories.allow_acute_cams)
%     violators(v_dist_cos > pi*3/5 & distance > 5) = 1;
    violators(needed_v_dist_cos > pi/2 & distance > 5) = 1;
end
violators(~consider) = 0; 
% violators = violators + violators';
% build impossibility matrix
impossibilityMatrix = zeros(length(trackletData));
impossibilityMatrix(violators & merging ~=1) = 1;
impossibilityMatrix(overlapping & merging ~=1) = 1;
impossibilityMatrix(~consider) = impossibilityMatrix(consider_nt); 
impossibilityMatrix(logical(eye(length(trackletData)))) = 0;
%% spacetimeAffinity
params = opts.trajectories;
% spacetimeAffinity = - params.weightDistance * distanceLoss + params.weightShapeChange * shapeChangeLoss - params.weightVelocityChange * velocityChangeLoss + params.weightIOU * (iouAffinity-0);
spacetimeAffinity = -0.5 * needed_v_dist_cos;
spacetimeAffinity(~consider) = spacetimeAffinity(consider_nt);
end


function overlap = overlapTest(interval1, interval2)

duration1       = interval1(2) - interval1(1);
duration2       = interval2(:,2) - interval2(:, 1);

i1              = repmat(interval1,size(interval2,1), 1);
unionMin        = min([i1, interval2], [], 2);
unionMax        = max([i1, interval2], [], 2);

overlap         = double(duration1 + duration2 - unionMax + unionMin >= 0);

end

