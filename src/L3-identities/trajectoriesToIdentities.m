function identities = trajectoriesToIdentities( trajectories, labels )

identities = [];

uniqueLabels = unique(labels);

for i = 1:length(uniqueLabels)
    
    trajectoryIndices = find(labels == uniqueLabels(i));
    
    trajectory =  struct('trajectories',[],'startFrame',inf,'endFrame',-inf);
    
    for k = 1:length(trajectoryIndices)
        ind = trajectoryIndices(k);
        trajectory.trajectories = [trajectory.trajectories; trajectories(ind)];
        trajectory.startFrame = min(trajectory.startFrame, trajectories(ind).startFrame);
        trajectory.endFrame = max(trajectory.startFrame, trajectories(ind).endFrame);
%         trajectory.segmentStart = min(trajectory.segmentStart, trajectories(ind).segmentStart);
%         trajectory.segmentEnd = max(trajectory.segmentEnd, trajectories(ind).segmentEnd);
%         trajectory.feature = trajectories(ind).feature;
    end
    
    identities = [identities; trajectory];
end

