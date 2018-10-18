function [newEmbedding,newDet,spatialGroupID_max] = createTracklets(opts, originalDetections, allFeatures, iCam, startFrame, endFrame,newEmbedding,newDet,spatialGroupID_max)
% CREATETRACKLETS This function creates short tracks composed of several detections.
%   In the first stage our method groups detections into space-time groups.
%   In the second stage a Binary Integer Program is solved for every space-time
%   group.

%% DIVIDE DETECTIONS IN SPATIAL GROUPS
% Initialize variables
params          = opts.tracklets;

% Find detections for the current frame interval
currentDetectionsIDX    = intervalSearch(originalDetections(:,1), startFrame, endFrame);

% Skip if no more than 1 detection are present in the scene
if length(currentDetectionsIDX) < 2, return; end

% Compute bounding box centeres
detectionCenters        = getBoundingBoxCenters(originalDetections(currentDetectionsIDX, 3:6)); 
detectionFrames         = originalDetections(currentDetectionsIDX, 1);

% Estimate velocities
estimatedVelocity       = estimateVelocities(originalDetections, startFrame, endFrame, params.nearest_neighbors, params.speed_limit);

% Spatial groupping
spatialGroupIDs         = getSpatialGroupIDs(opts.use_groupping, currentDetectionsIDX, detectionCenters, params)+spatialGroupID_max;


res = [ones(size(detectionFrames))*iCam,zeros(size(detectionFrames)),detectionFrames,spatialGroupIDs,detectionCenters,estimatedVelocity,zeros(size(detectionFrames)),cell2mat(allFeatures.appearance)];
newEmbedding = [newEmbedding;res];
newDet=[newDet;originalDetections];
spatialGroupID_max = max(spatialGroupIDs);
end


