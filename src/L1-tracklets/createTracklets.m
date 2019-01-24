function  tracklets = createTracklets(opts, originalDetections, allFeatures, startFrame, endFrame, tracklets,hyper_score_param)
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

iCam = unique(originalDetections(:,2));
if length(params.threshold)==8
    threshold = params.threshold(iCam);
else
    threshold = params.threshold;
end

if length(params.diff_p)==8
    diff_p = params.diff_p(iCam);
    diff_n = params.diff_n(iCam);
else
    diff_p = params.diff_p;
    diff_n = params.diff_n;
end


    %% SOLVE A GRAPH PARTITIONING PROBLEM FOR EACH SPATIAL GROUP
fprintf('Creating tracklets: solving space-time groups ');
for spatialGroupID = 1 : max(spatialGroupIDs)
    
    elements = find(spatialGroupIDs == spatialGroupID);
    spatialGroupObservations        = currentDetectionsIDX(elements);
    spatialGroupDetectionCenters    = detectionCenters(elements,:);
    spatialGroupDetectionFrames     = detectionFrames(elements,:);
    spatialGroupEstimatedVelocity   = estimatedVelocity(elements,:);
    
    if params.compute_score
        % Create an appearance affinity matrix and a motion affinity matrix
        appearanceCorrelation = getAppearanceSubMatrix(spatialGroupObservations, allFeatures, threshold,diff_p,diff_n,params.step);
    else
        features = cell2mat(allFeatures.appearance(spatialGroupObservations));
        appearanceCorrelation = getHyperScore(features,hyper_score_param,opts.soft, threshold,diff_p,0);
    end
    [motionCorrelation, impMatrix]  = motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,params.speed_limit, params.beta);
    
    % Combine affinities into correlations
    intervalDistance                = pdist2(spatialGroupDetectionFrames,spatialGroupDetectionFrames);
    discountMatrix                  = min(1, -log(intervalDistance/params.window_width));
%     correlationMatrix               = motionCorrelation + appearanceCorrelation; 
%     correlationMatrix               = correlationMatrix .* discountMatrix;
    correlationMatrix               = params.alpha*motionCorrelation .* discountMatrix + appearanceCorrelation; 
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
end


