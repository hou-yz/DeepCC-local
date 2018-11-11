function correlationMatrix = getHyperScore(features,tracklets,hyper_score_param,alpha)
tic;
if iscell(features)
    numFeatures = length(features);
    features = reshape(cell2mat(features)',256,[])';
else
    numFeatures = size(features,1);
end
numTracklets = length(tracklets);


input = repmat(reshape(features,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(features,numFeatures,1,[]),1,numFeatures,1);
input = abs(reshape(input,numFeatures*numFeatures,[]));


% if numTracklets == numFeatures && alpha 
% [~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);
% centerFrame     = round(mean(intervals,2));
% centers         = 0.5 * (endpoint + startpoint);
% frameDifference = pdist2(centerFrame, centerFrame, @(frame1, frame2) frame1 - frame2);
% % centersDistanceX = pdist2(centers(:,1),centers(:,1));
% % centersDistanceY = pdist2(centers(:,2),centers(:,2));
% velocityX       = repmat(velocity(:, 1), 1, numTracklets);
% velocityY       = repmat(velocity(:, 2), 1, numTracklets);
% startX          = repmat(centers(:, 1), 1, numTracklets);
% startY          = repmat(centers(:, 2), 1, numTracklets);
% endX            = repmat(centers(:, 1), 1, numTracklets);
% endY            = repmat(centers(:, 2), 1, numTracklets);
% errorXForward   = endX + velocityX .* frameDifference - startX';
% errorYForward   = endY + velocityY .* frameDifference - startY';
% errorXBackward  = startX' + velocityX' .* -frameDifference - endX;
% errorYBackward  = startY' + velocityY' .* -frameDifference - endY;
% errorForward    = sqrt(errorXForward.^2 + errorYForward.^2);
% errorBackward   = sqrt(errorXBackward.^2 + errorYBackward.^2);
% errorMatrix     = min(errorForward, errorBackward)/2203;%L2 norm of [1920,1080]
% input = [input,reshape(errorMatrix,numFeatures*numFeatures,[])];
% end


correlationMatrix = hyper_score_net_regress(input,hyper_score_param,alpha);
correlationMatrix = reshape(correlationMatrix,numFeatures,numFeatures);

t1=toc;
fprintf('time elapsed: %d',t1)
end

function output = hyper_score_net_regress(input,hyper_score_param,alpha)
feat = input(:,1:256);
% feat = feat.^2;
% if alpha
%     motion_score = input(:,257);
% end
% output=0;
output = max(feat*double(hyper_score_param.fc1_w')+double(hyper_score_param.fc1_b),0);
output = max(output*double(hyper_score_param.fc2_w')+double(hyper_score_param.fc2_b),0);
output = max(output*double(hyper_score_param.fc3_w')+double(hyper_score_param.fc3_b),0);
output = output*double(hyper_score_param.out_w')+double(hyper_score_param.out_b);
output = softmax(output')';
% if alpha
% appear_score = output(:,2);
% X = [ones(size(appear_score)),appear_score,motion_score,appear_score.*motion_score,appear_score.^2,motion_score.^2];
% b = [-1.0099;3.0465;-0.0486;-1.5418;-1.0273;0.0155];
% output = X*b;
% else
output = 2*output(:,2)-1;
% end

end
