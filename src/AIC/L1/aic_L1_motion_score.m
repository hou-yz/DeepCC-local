function [stAffinity, impossibilityMatrix] = aic_L1_motion_score(bboxs, world_pos, frames, speedLimit, beta)
centers = world_pos;

ious = bboxOverlapRatio(bboxs,bboxs);
iou_min = bboxOverlapRatio(bboxs,bboxs,'min');

frameDifference = repmat(frames,1,numel(frames)) - repmat(frames',numel(frames),1);
overlapping     = frameDifference == 0;
centersDistance = pdist2(centers,centers);
merging         = overlapping & (iou_min > 0);
% merging         = false;
velocity        = centersDistance./(abs(frameDifference)+10^-12)*10;
violators       = velocity > speedLimit;

% build impossibility matrix
impossibilityMatrix = zeros(numel(frames));
impossibilityMatrix(violators > 0 & merging ~=1) = 1;
impossibilityMatrix(overlapping == 1 & merging ~=1) = 1;


% compute space-time affinities
stAffinity      = ious-0.1*centersDistance;
end

