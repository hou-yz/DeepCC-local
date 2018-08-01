function spatialGroupIDs = getSpatialGroupIDs(useGrouping, currentDetectionsIDX, detectionCenters, params )
% Perfroms spatial groupping of detections and returns a vector of IDs

spatialGroupIDs         = ones(length(currentDetectionsIDX), 1);
if useGrouping == true
    
    pairwiseDistances   = pdist2(detectionCenters, detectionCenters);
    agglomeration       = linkage(pairwiseDistances);
    numSpatialGroups    = round(params.cluster_coeff * length(currentDetectionsIDX) / params.window_width);
    numSpatialGroups    = max(numSpatialGroups, 1);
    
    while true

        spatialGroupIDs     = cluster(agglomeration, 'maxclust', numSpatialGroups);
        uid = unique(spatialGroupIDs);
        freq = [histc(spatialGroupIDs(:),uid)];
        
        largestGroupSize = max(freq);
        % The BIP solver might run out of memory for large graphs
        if largestGroupSize <= 150 
            return
        end
        
        numSpatialGroups = numSpatialGroups + 1;
        
    end
    
end



