function trackletsInWindow = findTrackletsInWindow(trajectories, startTime, endTime)
trackletsInWindow = [];

if isempty(trajectories), return; end

trackletStartFrame = cell2mat({trajectories.startFrame});
trackletEndFrame   = cell2mat({trajectories.endFrame});
trackletsInWindow  = find( (trackletEndFrame >= startTime) .* (trackletStartFrame <= endTime) );

