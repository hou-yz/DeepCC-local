function trajectoriesInWindow = findTrajectoriesInWindow(trajectories, startTime, endTime)
trajectoriesInWindow = [];

if isempty(trajectories), return; end

trajectoryStartFrame = [trajectories.startFrame]; %cell2mat({trajectories.startFrame});
trajectoryEndFrame   = [trajectories.endFrame]; % cell2mat({trajectories.endFrame});
trajectoriesInWindow  = find( (trajectoryEndFrame >= startTime) .* (trajectoryStartFrame <= endTime) );

