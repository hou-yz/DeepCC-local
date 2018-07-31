function trajectories = trackletsToTrajectories( tracklets, labels )

trajectories = [];

uniqueLabels = unique(labels);

for i = 1:length(uniqueLabels)
    
    trackletIndices = find(labels == uniqueLabels(i));
    
    trajectory =  struct('tracklets',[],'startFrame',inf,'endFrame',-inf,'segmentStart',inf,'segmentEnd',-inf);
    
    for k = 1:length(trackletIndices)
        ind = trackletIndices(k);
        trajectory.tracklets = [trajectory.tracklets; tracklets(ind)];
        trajectory.startFrame = min(trajectory.startFrame, tracklets(ind).startFrame);
        trajectory.endFrame = max(trajectory.startFrame, tracklets(ind).endFrame);
        trajectory.segmentStart = min(trajectory.segmentStart, tracklets(ind).segmentStart);
        trajectory.segmentEnd = max(trajectory.segmentEnd, tracklets(ind).segmentEnd);
        trajectory.feature = tracklets(ind).feature;
    end
    
    trajectories = [trajectories; trajectory];
end

