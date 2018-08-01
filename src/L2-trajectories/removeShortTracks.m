function [ detectionsUpdated ] = removeShortTracks( detections, cutoffLength )
%This function removes short tracks that have not been associated with any
%trajectory. Those are likely to be false positives.

detectionsUpdated = detections;

detections = sortrows(detections, [2, 1]);

personIDs = unique(detections(:,2));
lengths = hist(detections(:,2), personIDs)';

[~,~, removedIDs] = find( personIDs .* (lengths <= cutoffLength));

detectionsUpdated(ismember(detectionsUpdated(:,2),removedIDs),:) = [];

