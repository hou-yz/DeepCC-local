function correlationMatrix = getHyperScore(features,tracklets,hyper_score_param,alpha,soft,threshold, norm)
tic;
if iscell(features)
    numFeatures = length(features);
    features = reshape(cell2mat(features)', numel(features{1}),[])';
else
    numFeatures = size(features,1);
end
numTracklets = length(tracklets);


input = repmat(reshape(features,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(features,numFeatures,1,[]),1,numFeatures,1);
input = abs(reshape(input,numFeatures*numFeatures,[]));

correlationMatrix = hyper_score_net_regress(input,hyper_score_param,alpha,soft,threshold, norm);
correlationMatrix = reshape(correlationMatrix,numFeatures,numFeatures);

t1=toc;
fprintf('time elapsed: %d',t1)
end

function output = hyper_score_net_regress(input,hyper_score_param,alpha,soft,threshold, norm)
feat = input;
% feat = feat.^2;
% if alpha
%     motion_score = input(:,257);
% end
% output=0;
output = max(feat*double(hyper_score_param.fc1_w')+double(hyper_score_param.fc1_b),0);
output = max(output*double(hyper_score_param.fc2_w')+double(hyper_score_param.fc2_b),0);
output = max(output*double(hyper_score_param.fc3_w')+double(hyper_score_param.fc3_b),0);
output = output*double(hyper_score_param.out_w')+double(hyper_score_param.out_b);
output = softmax(soft*output')';
% if alpha
% appear_score = output(:,2);
% X = [ones(size(appear_score)),appear_score,motion_score,appear_score.*motion_score,appear_score.^2,motion_score.^2];
% b = [-1.0099;3.0465;-0.0486;-1.5418;-1.0273;0.0155];
% output = X*b;
% else
output = (output(:,2)-output(:,1)-threshold)/norm;
% end

end
