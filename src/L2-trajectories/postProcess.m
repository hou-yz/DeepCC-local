function [trackerOutputFinal] = postProcess(dataset, extendedTracklets, trackerOutputFinal)
if isempty(extendedTracklets), return; end

lastID = 0;
if ~isempty(trackerOutputFinal), lastID = max(trackerOutputFinal(:, 1)); end

% collect tracker output
trackerOutputRaw = [];
for i = 1 : length(extendedTracklets)
    currentTrackletData = extendedTracklets(i).data;
    currentTrackletData(:, 1) = lastID + 1;
    lastID = lastID + 1;
    trackerOutputRaw = [trackerOutputRaw; currentTrackletData];
end

% interpolate to recover missing observations
trackerOutputFilled = fillTrajectories(trackerOutputRaw);

% remove very short trajectories
trackerOutputRemoved = removeShortTracks(trackerOutputFilled, dataset.minimumTrajectoryDuration);
%[~, ~, ic] = unique(trackerOutputRemoved(:,1));
%trackerOutputRemoved(:,1) = ic;
%trackerOutput = sortrows(trackerOutputRemoved,[2 1]);

trackerOutputFinal = sortrows(vertcat(trackerOutputFinal, trackerOutputRemoved), [2 1]);
