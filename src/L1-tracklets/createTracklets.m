function  tracklets = createTracklets(opts, originalDetections, allFeatures, startFrame, endFrame, tracklets)
% CREATETRACKLETS This function creates short tracks composed of several detections.
%   In the first stage our method groups detections into space-time groups.
%   In the second stage a Binary Integer Program is solved for every space-time
%   group.

%% DIVIDE DETECTIONS IN SPATIAL GROUPS
% Initialize variables
params          = opts.tracklets;
totalLabels     = 0; currentInterval = 0;

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
spatialGroupIDs         = getSpatialGroupIDs(opts.use_groupping, currentDetectionsIDX, detectionCenters, params);

% Show window detections
if opts.visualize, trackletsVisualizePart1; end

%% SOLVE A GRAPH PARTITIONING PROBLEM FOR EACH SPATIAL GROUP
fprintf('Creating tracklets: solving space-time groups ');
for spatialGroupID = 1 : max(spatialGroupIDs)
    
    elements = find(spatialGroupIDs == spatialGroupID);
    spatialGroupObservations        = currentDetectionsIDX(elements);
    
    % Create an appearance affinity matrix and a motion affinity matrix
    appearanceCorrelation           = getAppearanceSubMatrix(spatialGroupObservations, allFeatures, params.threshold);
    spatialGroupDetectionCenters    = detectionCenters(elements,:);
    spatialGroupDetectionFrames     = detectionFrames(elements,:);
    spatialGroupEstimatedVelocity   = estimatedVelocity(elements,:);
    [motionCorrelation, impMatrix]  = motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,params.speed_limit, params.beta);
    
    % Combine affinities into correlations
    intervalDistance                = pdist2(spatialGroupDetectionFrames,spatialGroupDetectionFrames);
    discountMatrix                  = min(1, -log(intervalDistance/params.window_width));
%     correlationMatrix               = motionCorrelation + appearanceCorrelation; 
%     correlationMatrix               = correlationMatrix .* discountMatrix;
    correlationMatrix               = motionCorrelation .* discountMatrix + appearanceCorrelation; 
    correlationMatrix(impMatrix==1) = -inf;
    
    % Show spatial grouping and correlations
    % if opts.visualize, trackletsVisualizePart2; end
    
    % Solve the graph partitioning problem
    fprintf('%d ',spatialGroupID);
    
    if strcmp(opts.optimization,'AL-ICM')
        labels = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        labels = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        labels = BIPCC(correlationMatrix, initialSolution);
    end
    
    labels      = labels + totalLabels;
    totalLabels = max(labels);
    identities  = labels;
    originalDetections(spatialGroupObservations, 2) = identities;
    
    % Show clustered detections
    if opts.visualize, trackletsVisualizePart3; end
end
fprintf('\n');

%% FINALIZE TRACKLETS
% Fit a low degree polynomial to include missing detections and smooth the tracklet
trackletsToSmooth  = originalDetections(currentDetectionsIDX,:);
featuresAppearance = allFeatures.appearance(currentDetectionsIDX);
smoothedTracklets  = smoothTracklets(trackletsToSmooth, startFrame, params.window_width, featuresAppearance, params.min_length, currentInterval);

% Assign IDs to all tracklets
for i = 1:length(smoothedTracklets)
    smoothedTracklets(i).id = i;
    smoothedTracklets(i).ids = i;
end

% Attach new tracklets to the ones already discovered from this batch of detections
if ~isempty(smoothedTracklets)
    ids = 1 : length(smoothedTracklets); 
    tracklets = [tracklets, smoothedTracklets];
end

% Show generated tracklets in window
if opts.visualize, trackletsVisualizePart4; end

if ~isempty(tracklets)
    tracklets = nestedSortStruct(tracklets,{'startFrame','endFrame'});
end

