function imp_score = impossibility_frame_overlap(startFrames,endFrames)
%IMPOSSIBILITY_FRAME_OVERLAP.M Summary of this function goes here
%   Detailed explanation goes here
frame_dist1 = pdist2(startFrames,startFrames);
frame_dist2 = pdist2(endFrames,endFrames);
imp_score = (frame_dist1 + frame_dist2)==0;
imp_score(logical(eye(length(imp_score)))) = 0;
end

