function [stAffinity, impossibilityMatrix, indiffMatrix] = aic_L2_motion_score(opts, tracklets, beta, speedLimit, indifferenceLimit, iCam)

numTracklets = length(tracklets);

[~, ~, startpoint, endpoint, intervals, ~, velocity, bboxs] = getTrackletFeatures(tracklets);
centers         = 0.5 * (endpoint + startpoint);

centerFrame     = round(mean(intervals,2));
frameDifference = pdist2(centerFrame, centerFrame, @(frame1, frame2) frame1 - frame2);
overlapping     = pdist2(intervals,intervals, @overlapTest);
centersDistance = pdist2(centers,centers);
merging         = (centersDistance < 5) & overlapping;
maxSpeedMatrix  = centersDistance./abs(frameDifference);
violators       = maxSpeedMatrix > speedLimit;
violators(overlapping > 0) = 0;



forward_centers = centers + centerFrame .* velocity;
backward_centers = centers - centerFrame .* velocity;
forward_dist = pdist2(forward_centers,centers);
backward_dist = pdist2(backward_centers,centers);
forward_error = forward_dist ./ sqrt(sum(velocity.^2,2));
backward_error = backward_dist ./ sqrt(sum(velocity.^2,2));
% this is a symmetric matrix, although tracklets are oriented in time
errorMatrix     = min(forward_error, backward_error) > opts.trajectories.window_width;
% violators       = errorMatrix > opts.trajectories.window_width;
% violators(overlapping > 0) = 0;


forward_bboxs = bboxs;backward_bboxs = bboxs;
forward_bboxs(:,[1,2]) = bboxs(:,[1,2]) + centerFrame .* velocity;
backward_bboxs(:,[1,2]) = bboxs(:,[1,2]) - centerFrame .* velocity;
forward_ious = bboxOverlapRatio(forward_bboxs,bboxs);
backward_ious = bboxOverlapRatio(backward_bboxs,bboxs);
ious = forward_ious + backward_ious;

speed_cosine_dist = pdist2(velocity,velocity,'cosine');

% compute space-time affinities
% stAffinity      = 1 - beta*errorMatrix + ious;
stAffinity      = 0 - speed_cosine_dist - errorMatrix - (ious == 0);


% build impossibility matrix
impossibilityMatrix = zeros(numTracklets);
% impossibilityMatrix(violators > 0 & merging ~=1) = 1;
% impossibilityMatrix(overlapping == 1 & merging ~=1) = 1;


% compute indifference matrix
timeDifference  = frameDifference .* (frameDifference > 0);
timeDifference  = timeDifference + timeDifference';
indiffMatrix    = 1 - sigmf(timeDifference,[0.1 indifferenceLimit/2]);
end


function overlap = overlapTest(interval1, interval2)

duration1       = interval1(2) - interval1(1);
duration2       = interval2(:,2) - interval2(:, 1);

i1              = repmat(interval1,size(interval2,1), 1);
unionMin        = min([i1, interval2], [], 2);
unionMax        = max([i1, interval2], [], 2);

overlap         = double(duration1 + duration2 - unionMax + unionMin >= 0);

end
