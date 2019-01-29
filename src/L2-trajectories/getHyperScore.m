function correlationMatrix = getHyperScore(features,model_param,soft,threshold, norm, motion)
tic;
if iscell(features)
    numFeatures = length(features);
    features = reshape(cell2mat(features)', numel(features{1}),[])';
else
    numFeatures = size(features,1);
end

if ~motion
    data = repmat(reshape(features,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(features,numFeatures,1,[]),1,numFeatures,1);
    data = abs(reshape(data,numFeatures*numFeatures,[]));
else
    centerFrame = features(:,1);
    centers = features(:,[2,3]);
    velocity = features(:,[4,5]);
    
    feat1 = [centerFrame,-centers,zeros(size(centers)),-velocity,zeros(size(velocity))];
    feat2 = [centerFrame,zeros(size(centers)),centers,zeros(size(velocity)),velocity];
    data = repmat(reshape(feat2,1,numFeatures,[]),numFeatures,1,1) - repmat(reshape(feat1,numFeatures,1,[]),1,numFeatures,1);
    bad_lines = logical(tril(ones(length(centerFrame)),-1));
    
    data = reshape(data,numFeatures*numFeatures,[]);
    data(:,6:9) = data(:,6:9).*data(:,1);
    data(:,1) = [];
end

correlationMatrix = hyper_score_net(data,model_param,soft);
correlationMatrix = reshape(correlationMatrix,numFeatures,numFeatures);

if motion
    transposeMatrix = correlationMatrix.';
    correlationMatrix(bad_lines) = transposeMatrix(bad_lines);
end

t1=toc;
fprintf('time elapsed: %d',t1)
end

function output = hyper_score_net(data,model_param,soft)
feat = data;
output = max(feat*double(model_param.fc1_w')+double(model_param.fc1_b),0);
output = max(output*double(model_param.fc2_w')+double(model_param.fc2_b),0);
output = max(output*double(model_param.fc3_w')+double(model_param.fc3_b),0);
output = output*double(model_param.out_w')+double(model_param.out_b);
output = softmax(soft*output')';
output = (output(:,2)-output(:,1))/soft;

end
