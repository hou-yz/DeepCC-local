function [ mergedTracklets ] = mergeTracklets( tracklets, labels )
% This function merges co-identified tracklets

uniqueLabels = unique(labels);

for i = 1:length(uniqueLabels)
    
    currentLabel = uniqueLabels(i);
    
    trackletIndicesToMerge = find(labels == currentLabel);
    
    newTracklet = [];
    newTracklet.startFrame = inf;
    newTracklet.endFrame = -1;
    newTracklet.ids = [];
    newTracklet.data = [];
    newTracklet.realdata = [];
    newTracklet.feature = [];
    newTracklet.features = [];
    newTracklet.mask = [];
    newTracklet.id = 0;
    
    for k = 1:length(trackletIndicesToMerge)
        
        id = trackletIndicesToMerge(k);
        
        newTracklet.data = [newTracklet.data; tracklets(id).data];
        newTracklet.data = sortrows(newTracklet.data,2);
        
        
        newTracklet.realdata    = [newTracklet.realdata; tracklets(id).realdata];
        newTracklet.features    = [newTracklet.features; tracklets(id).features];
        newTracklet.startFrame  = min(newTracklet.startFrame, tracklets(id).startFrame);
        newTracklet.endFrame    = max(newTracklet.endFrame, tracklets(id).endFrame);
        newTracklet.ids         = [newTracklet.ids; tracklets(id).ids];
        newTracklet.interval    =  tracklets(id).interval;
        newTracklet.center      =  tracklets(id).center;
        newTracklet.centerWorld =  tracklets(id).centerWorld;
        
    end
    
    % Re-compute the appearance descriptor
    medianFeature = median(cell2mat(newTracklet.features));
    newTracklet.feature = medianFeature;
    
    mergedTracklets(i) = newTracklet;
    
end

mergedTracklets = mergedTracklets';
