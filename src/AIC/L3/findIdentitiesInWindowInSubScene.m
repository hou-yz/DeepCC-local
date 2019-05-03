function trajectoriesInWindow = findIdentitiesInWindowInSubScene(ids, startTime, endTime, cams_in_subscene)
trajectoriesInWindow = [];

if isempty(ids), return; end

in_subscene = zeros(1,length(ids));
for i = 1:length(ids)
    in_subscene(i) = sum(ismember(ids(i).iCams,cams_in_subscene));
end

trajectoryStartFrame = [ids.startFrame]; %cell2mat({trajectories.startFrame});
trajectoryEndFrame   = [ids.endFrame]; % cell2mat({trajectories.endFrame});
trajectoriesInWindow  = find( (trajectoryEndFrame >= startTime) .* (trajectoryStartFrame <= endTime) .* in_subscene );

end