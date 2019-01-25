function correlationMatrix = getHyperScore(features,model_param,soft,threshold, norm, motion)
tic;
if iscell(features)
    numFeatures = length(features);
    features = reshape(cell2mat(features)', numel(features{1}),[])';
else
    numFeatures = size(features,1);
end

if ~motion
    input = repmat(reshape(features,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(features,numFeatures,1,[]),1,numFeatures,1);
    input = abs(reshape(input,numFeatures*numFeatures,[]));
else
    feat1 = features;
    feat2 = [features(:,1),-features(:,2:end)];
    input = repmat(reshape(feat1,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(feat2,numFeatures,1,[]),1,numFeatures,1);
    input = reshape(input,numFeatures*numFeatures,[]);
    input(:,1) = abs(input(:,1));
    input(:,6:9) = input(:,6:9).*input(:,1);
    input(:,1) = [];
end

correlationMatrix = hyper_score_net(input,model_param,soft,threshold, norm);
correlationMatrix = reshape(correlationMatrix,numFeatures,numFeatures);

t1=toc;
fprintf('time elapsed: %d',t1)
end

function output = hyper_score_net(input,model_param,soft,threshold, norm)
feat = input;
output = max(feat*double(model_param.fc1_w')+double(model_param.fc1_b),0);
output = max(output*double(model_param.fc2_w')+double(model_param.fc2_b),0);
output = max(output*double(model_param.fc3_w')+double(model_param.fc3_b),0);
output = output*double(model_param.out_w')+double(model_param.out_b);
output = softmax(soft*output')';
output = (output(:,2)-output(:,1)-threshold)/norm;

end
