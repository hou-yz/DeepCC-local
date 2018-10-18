function tracklets = solveTracklets(opts, originalDetections, newEmbedding, correlationMatrix_source, iCam, startFrame, endFrame, tracklets)
% Initialize variables
params          = opts.tracklets;
totalLabels     = 0; currentInterval = 0;
% Find detections for the current frame interval
currentDetectionsIDX    = intervalSearch(originalDetections(:,1), startFrame, endFrame);
% Skip if no more than 1 detection are present in the scene
if length(currentDetectionsIDX) < 2, return; end

detectionCenters = newEmbedding(currentDetectionsIDX,5:6);
detectionFrames = newEmbedding(currentDetectionsIDX,3);
estimatedVelocity = newEmbedding(currentDetectionsIDX,7:8);
allFeatures = newEmbedding(currentDetectionsIDX,10:end);
spatialGroupIDs = newEmbedding(currentDetectionsIDX,4);

% Show window detections
if opts.visualize, trackletsVisualizePart1; end

% iCam = originalDetections(1,2);
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
for spatialGroupID = min(spatialGroupIDs) : max(spatialGroupIDs)
    
    elements = find(spatialGroupIDs == spatialGroupID);
    spatialGroupObservations        = currentDetectionsIDX(elements);
    l = length(spatialGroupObservations);
    
    if params.compute_score
    % Create an appearance affinity matrix and a motion affinity matrix
    appearanceCorrelation           = getAppearanceSubMatrix(spatialGroupObservations, allFeatures, threshold,diff_p,diff_n,params.step);
    spatialGroupDetectionCenters    = detectionCenters(elements,:);
    spatialGroupDetectionFrames     = detectionFrames(elements,:);
    spatialGroupEstimatedVelocity   = estimatedVelocity(elements,:);
    [motionCorrelation, impMatrix]  = motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,params.speed_limit, params.beta);
    
    % Combine affinities into correlations
    intervalDistance                = pdist2(spatialGroupDetectionFrames,spatialGroupDetectionFrames);
    discountMatrix                  = min(1, -log(intervalDistance/params.window_width));
%     correlationMatrix               = motionCorrelation + appearanceCorrelation; 
%     correlationMatrix               = correlationMatrix .* discountMatrix;
    correlationMatrix               = params.alpha*motionCorrelation .* discountMatrix + appearanceCorrelation; 
    correlationMatrix(impMatrix==1) = -inf;
    else
        elements = find(correlationMatrix_source(:,1) == spatialGroupID);
        correlationMatrix        = reshape(correlationMatrix_source(elements,2),[l,l]);
%         spatialGroupDetectionCenters    = detectionCenters(elements,:);
%         spatialGroupDetectionFrames     = detectionFrames(elements,:);
%         spatialGroupEstimatedVelocity   = estimatedVelocity(elements,:);
%         [motionCorrelation, impMatrix]  = motionAffinity(spatialGroupDetectionCenters,spatialGroupDetectionFrames,spatialGroupEstimatedVelocity,params.speed_limit, params.beta);
%         correlationMatrix(impMatrix==1) = -inf;
        
    end
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
% featuresAppearance = allFeatures.appearance(currentDetectionsIDX);
featuresAppearance = allFeatures(currentDetectionsIDX,:);
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

