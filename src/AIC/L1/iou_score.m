function [ious] = iou_score(bboxs)
%iou_score Summary of this function goes here
%   Detailed explanation goes here
ious = bboxOverlapRatio(bboxs,bboxs);
ious(ious == 0) = -1;
ious = ious/2;
% ious(ious>0) = ious(ious>0)/0.8;
% ious(ious<=0) = ious(ious<=0)/0.2;
end

