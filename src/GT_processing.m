function  [appendedGTs,spatialGroupID_max] = GT_processing(opts, originalGTs, startFrame, endFrame, spatialGroupID_max)
% CREATETRACKLETS This function creates short tracks composed of several detections.
%   In the first stage our method groups detections into space-time groups.
%   In the second stage a Binary Integer Program is solved for every space-time
%   group.

%% DIVIDE DETECTIONS IN SPATIAL GROUPS
% Initialize variables
params          = opts.tracklets;
totalLabels     = 0; currentInterval = 0;

% Find detections for the current frame interval
currentGTsIDX    = intervalSearch(originalGTs(:,1), startFrame, endFrame);

appendedGTs=[];
% Skip if no more than 1 detection are present in the scene
if isempty(currentGTsIDX)
    return; 
end

assert(length(currentGTsIDX) == size(originalGTs,1),'miss in GTs')




% add bbox position jitter before extracting center & speed
jitter_max = 0.03*originalGTs(currentGTsIDX,5:6);
originalGTs(currentGTsIDX,3:4) = originalGTs(currentGTsIDX,3:4)+jitter_max.*(rand(size(jitter_max))-0.5);
originalGTs(currentGTsIDX,5:6) = originalGTs(currentGTsIDX,5:6)+jitter_max.*(rand(size(jitter_max))-0.5);

% Compute bounding box centeres
gtCenters        = getBoundingBoxCenters(originalGTs(currentGTsIDX, 3:6)); 
gtFrames         = originalGTs(currentGTsIDX, 1);

% Estimate velocities
estimatedVelocity       = estimateVelocities(originalGTs, startFrame, endFrame, params.nearest_neighbors, params.speed_limit);

% Spatial groupping
if length(currentGTsIDX)>1
    spatialGroupIDs = getSpatialGroupIDs(opts.use_groupping, currentGTsIDX, gtCenters, params);
else
    spatialGroupIDs = [1];
end
spatialGroupIDs = spatialGroupIDs+spatialGroupID_max;
spatialGroupID_max = max(spatialGroupIDs);
% Show window detections
if opts.visualize, trackletsVisualizePart1; end

appendedGTs = [originalGTs(:,1:2),spatialGroupIDs,gtCenters,estimatedVelocity];

end
