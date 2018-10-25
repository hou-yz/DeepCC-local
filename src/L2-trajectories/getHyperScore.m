function correlationMatrix = getHyperScore(opts,features,hyper_score_param)
tic;
numFeatures = length(features);
feat = reshape(cell2mat(features)',256,[])';

% [~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);
% centerFrame     = round(mean(intervals,2));
% centers         = 0.5 * (endpoint + startpoint);
% 
% pids = zeros(numFeatures,1);
% newGTs = [ones(numFeatures,1)*iCam,pids,centerFrame,zeros(numFeatures,1),centers,velocity,zeros(numFeatures,1),feat];
% newGTs = newGTs(:,[1,3,5:8,10:end]);
% a = [newGTs(:,1:4),zeros(numFeatures,2),newGTs(:,5:end)];
% b = [newGTs(:,1:6),zeros(numFeatures,2),newGTs(:,7:end)];

input = repmat(reshape(feat,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(feat,numFeatures,1,[]),1,numFeatures,1);
input = abs(reshape(input,numFeatures*numFeatures,[]));
% input = [input(:,1:8),vecnorm(input(:,9:end),2,2)];
out = hyper_score_net(input,hyper_score_param);
correlationMatrix = (out(:,2)-0.5)*2;
correlationMatrix = reshape(correlationMatrix,numFeatures,numFeatures);

t1=toc;
fprintf('time elapsed: %d',t1)
end

function output = hyper_score_net(input,hyper_score_param)
output = max(input*double(hyper_score_param.fc1_w')+double(hyper_score_param.fc1_b),0);
output = max(output*double(hyper_score_param.fc2_w')+double(hyper_score_param.fc2_b),0);
output = max(output*double(hyper_score_param.fc3_w')+double(hyper_score_param.fc3_b),0);
output = output*double(hyper_score_param.out_w')+double(hyper_score_param.out_b);
output = softmax(output')';
end



