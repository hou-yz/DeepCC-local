function trajectories = trackletsToTrajectories(opts, tracklets, labels )

trajectories = [];

uniqueLabels = unique(labels);

for i = 1:length(uniqueLabels)
    
    trackletIndices = find(labels == uniqueLabels(i));
    
    trajectory =  struct('tracklets',[],'startFrame',inf,'endFrame',-inf,'segmentStart',inf,'segmentEnd',-inf,'features',[]);
    for k = 1:length(trackletIndices)
        ind = trackletIndices(k);
        trajectory.tracklets = [trajectory.tracklets; tracklets(ind)];
        trajectory.startFrame = min(trajectory.startFrame, tracklets(ind).startFrame);
        trajectory.endFrame = max(trajectory.startFrame, tracklets(ind).endFrame);
        trajectory.segmentStart = min(trajectory.segmentStart, tracklets(ind).segmentStart);
        trajectory.segmentEnd = max(trajectory.segmentEnd, tracklets(ind).segmentEnd);
%         trajectory.feature = tracklets(ind).feature;
        trajectory.features = [trajectory.features;tracklets(ind).features];
    end
%     trajectory.feature = mean(cell2mat(trajectory.features));

    real_frame_length = 0;
    trajectory.feature=zeros(size(trajectory.tracklets(1).feature));
    for k = 1:length(trajectory.tracklets)
        trajectory.feature = trajectory.feature+trajectory.tracklets(k).feature.*length(trajectory.tracklets(k).realdata);
        real_frame_length = real_frame_length + length(trajectory.tracklets(k).realdata);
    end
    trajectory.feature = trajectory.feature/real_frame_length;
    
    trajectories = [trajectories; trajectory];
end

