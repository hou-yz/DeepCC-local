function [ious] = iouAffinity(bboxs,centers)
%IOUAFFINITY Summary of this function goes here
%   Detailed explanation goes here
ious = bboxOverlapRatio(bboxs,bboxs);
ious = ious - 0.2;
ious(ious>0) = ious(ious>0)/0.8;
ious(ious<=0) = ious(ious<=0)/0.2;

% center_dist_xs = pdist2(centers(:,1),centers(:,1));
% center_dist_ys = pdist2(centers(:,2),centers(:,2));
% bbox_size_xs = bboxs(:,1);
% bbox_size_ys = bboxs(:,2);
end

