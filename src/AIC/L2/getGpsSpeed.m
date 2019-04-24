function [ startpoint, endpoint, velocity,intervals] = getGpsSpeed( trackletData)

numTracklets = length(trackletData);

% calculate velocity, direction, for each tracklet

velocity = zeros(numTracklets,2);
startpoint = zeros(numTracklets,2);
endpoint = zeros(numTracklets,2);
duration = zeros(numTracklets,1);
intervals = zeros(numTracklets,2);


for ind = 1:numTracklets
        intervals(ind,:) = [trackletData{ind}(1,1), trackletData{ind}(end,1)];
        startpoint(ind,:) = [trackletData{ind}(1,7), trackletData{ind}(1,8)];
        endpoint(ind,:) = [trackletData{ind}(end,7), trackletData{ind}(end,8)];
        duration(ind) = intervals(ind,2) - intervals(ind,1);
        direction = endpoint(ind,:) - startpoint(ind,:);
        velocity(ind,:) = direction./duration(ind)*10;
end
end







