function newTrajectories = recomputeTrajectories( trajectories ,opts )
%RECOMPUTETRAJECTORIES Summary of this function goes here
%   Detailed explanation goes here

% newTrajectories = trajectories;

segmentLength = opts.tracklets.window_width;

if isempty(trajectories)
    newTrajectories = [];
    return
end

for i = 1:length(trajectories)
    segmentStart = trajectories(i).segmentStart;
    segmentEnd = trajectories(i).segmentEnd;
    numSegments = (segmentEnd + 1 - segmentStart) / segmentLength;
    
    alldata = {trajectories(i).tracklets(:).data};
    alldata = cell2mat(alldata');
    alldata = sortrows(alldata,2);
    [~, uniqueRows] = unique(alldata(:,1));
    alldata = alldata(uniqueRows,:);
    
    realdata = {trajectories(i).tracklets(:).realdata};
    realdata = cell2mat(realdata');
    realdata = sortrows(realdata,2);
    [~, uniqueRows] = unique(realdata(:,1));
    realdata = realdata(uniqueRows,:);
    
    dataFrames = alldata(:,1);
    
    frames = segmentStart:segmentEnd;
    interestingFrames = round([min(dataFrames), frames(1) + segmentLength/2:segmentLength:frames(end),  max(dataFrames)]);
        
    keyData = alldata(ismember(dataFrames,interestingFrames),:);    
    keyData(:,2) = -1;
    newData = fillTrajectories(keyData);
    newData = sortrows(newData);
    
    newTrajectory = trajectories(i);
    sampleTracklet = trajectories(i).tracklets(1);
    newTrajectory.tracklets = [];
    features = trajectories(i).features;
    feature = trajectories(i).feature;
    
    for k = 1:numSegments
        tracklet = sampleTracklet;
        tracklet.segmentStart = segmentStart + (k-1)*segmentLength;
        tracklet.segmentEnd   = tracklet.segmentStart + segmentLength - 1;
        
        trackletFrames = tracklet.segmentStart:tracklet.segmentEnd;
        rows = ismember(newData(:,1), trackletFrames);
        
        tracklet.data = newData(rows,:);
        
        tracklet.startFrame = min(tracklet.data(:,1));
        tracklet.endFrame = max(tracklet.data(:,1));
        tracklet.feature = feature;
        if k == 1
            tracklet.features = features;
            tracklet.realdata = realdata;
        else
            tracklet.features = [];
            tracklet.realdata = [];
        end
        newTrajectory.startFrame = min(newTrajectory.startFrame, tracklet.startFrame);
        newTrajectory.endFrame = max(newTrajectory.endFrame, tracklet.endFrame);
        
        newTrajectory.tracklets = [newTrajectory.tracklets; tracklet];
        
    end
    newTrajectories(i) = newTrajectory;
end

end
