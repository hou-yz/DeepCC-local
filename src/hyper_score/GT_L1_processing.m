function  [edge_weight,spatialGroup_max] = GT_L1_processing(opts, originalGTs, startFrame, endFrame, spatialGroup_max,use_spaGrp)
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
edge_weight = [];

% Skip if no more than 1 detection are present in the scene
if isempty(currentGTsIDX)
    return; 
end

assert(length(currentGTsIDX) == size(originalGTs,1),'miss in GTs')




% add bbox position jitter before extracting center & speed
bboxs = originalGTs(currentGTsIDX,3:6);
edge_weight = bboxOverlapRatio(bboxs,bboxs);


spatialGroupIDs = ones(size(currentGTsIDX,1),1);
spatialGroupIDs = spatialGroupIDs+spatialGroup_max;
spatialGroup_max = max(spatialGroupIDs);


%%
if opts.visualize
    detectionCenters = getBoundingBoxCenters(originalGTs(currentGTsIDX, 3:6));
    trackletsVisualizePart1
end
end
