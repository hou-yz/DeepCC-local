function [stAffinity, impossibilityMatrix] = aic_L1_motion_score(bboxs, frames, speedLimit, beta)
centers = bboxs(:,[1,2]) + 0.5*bboxs(:,[3,4]);

ious = bboxOverlapRatio(bboxs,bboxs);

frameDifference = repmat(frames,1,numel(frames)) - repmat(frames',numel(frames),1);
overlapping     = frameDifference == 0;
centersDistance = pdist2(centers,centers);
bbox_sizes      = sqrt(sum(bboxs(:,[3,4]).^2,2));
bbox_sizes      = min(bbox_sizes,bbox_sizes');
merging         = (centersDistance < 0.5*bbox_sizes) & overlapping & (ious > 0);
velocity        = centersDistance./abs(frameDifference);
violators       = velocity > speedLimit;
violators(overlapping > 0) = 0;

% build impossibility matrix
impossibilityMatrix = zeros(numel(frames));
impossibilityMatrix(violators > 0 & merging ~=1) = 1;
impossibilityMatrix(overlapping == 1 & merging ~=1) = 1;


% compute space-time affinities
% stAffinity      = 1 - beta*errorMatrix + ious;
stAffinity      = ious/2;
stAffinity(ious == 0) = -0.5;
end

