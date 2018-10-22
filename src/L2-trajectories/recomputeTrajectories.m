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
    dataFrames = alldata(:,1);
    
    frames = segmentStart:segmentEnd;
    interestingFrames = round([min(dataFrames), frames(1) + segmentLength/2:segmentLength:frames(end),  max(dataFrames)]);
        
    keyData = alldata(ismember(dataFrames,interestingFrames),:);
    
%     for k = size(keyData,1)-1:-1:1
%         
%         while keyData(k,2) == keyData(k+1,2)
%             keyData(k+1,:) = [];
%         end
%         
%     end
    
    keyData(:,2) = -1;
    newData = fillTrajectories(keyData);
    newData = sortrows(newData);
    
    newTrajectory = trajectories(i);
    sampleTracklet = trajectories(i).tracklets(1);
    newTrajectory.tracklets = [];
    
    
    for k = 1:numSegments
       
        tracklet = sampleTracklet;
        tracklet.segmentStart = segmentStart + (k-1)*segmentLength;
        tracklet.segmentEnd   = tracklet.segmentStart + segmentLength - 1;
        
        trackletFrames = tracklet.segmentStart:tracklet.segmentEnd;
        
        
        rows = ismember(newData(:,1), trackletFrames);
        
        tracklet.data = newData(rows,:);
        tracklet.realdata = [];
        
        tracklet.startFrame = min(tracklet.data(:,1));
        tracklet.endFrame = max(tracklet.data(:,1));
        
        
%         if isempty(tracklet.data)
%             tracklet.startFrame = min(tracklet.realdata(:,1));
%             tracklet.endFrame = max(tracklet.realdata(:,1));
%         end
        newTrajectory.startFrame = min(newTrajectory.startFrame, tracklet.startFrame);
        newTrajectory.endFrame = max(newTrajectory.endFrame, tracklet.endFrame);
        
        newTrajectory.tracklets = [newTrajectory.tracklets; tracklet];
        
    end
%     [~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(newTrajectory.tracklets);
%     centers = num2cell(0.5*(endpoint+startpoint),2);
%     [newTrajectory.tracklets.centers] = centers{:};
%     interval = num2cell(intervals,2);
%     [newTrajectory.tracklets.interval] = interval{:};
%     velocity = num2cell(velocity,2);
%     [newTrajectory.tracklets.velocity] = velocity{:};
    newTrajectories(i) = newTrajectory;
    

end

