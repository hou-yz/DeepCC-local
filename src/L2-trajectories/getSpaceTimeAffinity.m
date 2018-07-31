function [stAffinity, impossibilityMatrix, indiffMatrix] = getSpaceTimeAffinity(tracklets, beta1, speedLimit, indifferenceLimit)

numTracklets = length(tracklets);

[~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);

centerFrame     = round(mean(intervals,2));
frameDifference = pdist2(centerFrame, centerFrame, @(frame1, frame2) frame1 - frame2);
overlapping     = pdist2(intervals,intervals, @overlapTest);
centers         = 0.5 * (endpoint + startpoint);
centersDistance = pdist2(centers,centers);
v               = (frameDifference > 0) | overlapping;
merging         = (centersDistance < 5) & overlapping;

velocityX       = repmat(velocity(:, 1), 1, numTracklets);
velocityY       = repmat(velocity(:, 2), 1, numTracklets);

startX          = repmat(centers(:, 1), 1, numTracklets);
startY          = repmat(centers(:, 2), 1, numTracklets);
endX            = repmat(centers(:, 1), 1, numTracklets);
endY            = repmat(centers(:, 2), 1, numTracklets);

errorXForward   = endX + velocityX .* frameDifference - startX';
errorYForward   = endY + velocityY .* frameDifference - startY';

errorXBackward  = startX' + velocityX' .* -frameDifference - endX;
errorYBackward  = startY' + velocityY' .* -frameDifference - endY;

errorForward    = sqrt(errorXForward.^2 + errorYForward.^2);
errorBackward   = sqrt(errorXBackward.^2 + errorYBackward.^2);

% check if speed limit is violated
xDiff           = endX - startX';
yDiff           = endY - startY';
distanceMatrix  = sqrt(xDiff.^2 + yDiff.^2);
maxSpeedMatrix  = distanceMatrix./abs(frameDifference);

violators       = maxSpeedMatrix > speedLimit;
violators(~v)   = 0;
violators       = violators + violators';

% build impossibility matrix
impossibilityMatrix = zeros(numTracklets);
impossibilityMatrix(violators == 1 & merging ~=1) = 1;
impossibilityMatrix(overlapping == 1 & merging ~=1) = 1;

% this is a symmetric matrix, although tracklets are oriented in time
errorMatrix     = min(errorForward, errorBackward);
errorMatrix     = errorMatrix .* v;
errorMatrix(~v) = 0;
errorMatrix     = errorMatrix + errorMatrix';
errorMatrix(violators == 1) = Inf;

% compute indifference matrix
timeDifference  = frameDifference .* (frameDifference > 0);
timeDifference  = timeDifference + timeDifference';
indiffMatrix    = 1 - sigmf(timeDifference,[0.1 indifferenceLimit/2]);

% compute space-time affinities
stAffinity      = 1 - beta1*errorMatrix;
stAffinity      = max(0, stAffinity);

end

function overlap = overlapTest(interval1, interval2)

duration1       = interval1(2) - interval1(1);
duration2       = interval2(:,2) - interval2(:, 1);

i1              = repmat(interval1,size(interval2,1), 1);
unionMin        = min([i1, interval2], [], 2);
unionMax        = max([i1, interval2], [], 2);

overlap         = double(duration1 + duration2 - unionMax + unionMin >= 0);

end






