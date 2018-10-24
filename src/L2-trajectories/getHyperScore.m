function correlationMatrix = getHyperScore(opts,tracklets,features,iCam,hyper_score_param)
tic;
numTracklets = length(tracklets);
feat = reshape(cell2mat(features)',256,[])';

[~, ~, startpoint, endpoint, intervals, ~, velocity] = getTrackletFeatures(tracklets);
centerFrame     = round(mean(intervals,2));
centers         = 0.5 * (endpoint + startpoint);

pids = zeros(numTracklets,1);
newGTs = [ones(numTracklets,1)*iCam,pids,centerFrame,zeros(numTracklets,1),centers,velocity,zeros(numTracklets,1),feat];

% hdf5write(fullfile(opts.experiment_root,opts.experiment_name, 'L2-trajectories','hyperEMB_tmp.h5'), '/hyperGT',newGTs');
% 
% current_dir = pwd;
% 
% cd('src/hyper_score')
% commandStr = sprintf('CUDA_VISIBLE_DEVICES=5 %s main.py --data-path %s/experiments/%s/L2-trajectories/hyperEMB_tmp.h5 --save_result',opts.python,current_dir,opts.experiment_name);
% [status, commandOut] = system(commandStr);
% if status==0 
%      fprintf('result is %s\n',commandOut);
% end
% 
% cd(current_dir)
% res = h5read(fullfile(opts.experiment_root,opts.experiment_name, 'L2-trajectories','pairwise_dis_tmp.h5'),'/dis');
% res = reshape(res',numTracklets,numTracklets)';
% correlationMatrix = res+res';
newGTs = newGTs(:,[1,3,5:8,10:end]);
a = [newGTs(:,1:4),zeros(numTracklets,2),newGTs(:,5:end)];
b = [newGTs(:,1:6),zeros(numTracklets,2),newGTs(:,7:end)];
input = repmat(reshape(a,1,numTracklets,[]),numTracklets,1,1) - repmat(reshape(b,numTracklets,1,[]),1,numTracklets,1);
input(:,:,[5,6]) = -input(:,:,[5,6]);
input = abs(reshape(input,numTracklets*numTracklets,[]));
% input = [input(:,1:8),vecnorm(input(:,9:end),2,2)];
out = hyper_score_net(input,hyper_score_param);
correlationMatrix = (out(:,2)-0.5)*2;
correlationMatrix = reshape(correlationMatrix,numTracklets,numTracklets);

% index = find(triu(ones(numTracklets),1));
% correlationMatrix = zeros(numTracklets);
% correlationMatrix(index) = out(index);
% correlationMatrix = correlationMatrix+correlationMatrix';

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



