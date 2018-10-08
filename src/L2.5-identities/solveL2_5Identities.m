function [result,correlation] = solveL2_5Identities(opts,source_traj,source_label, target_trajs, target_labels)
% same id threshold
threshold=0.3;

params = opts.identities;
% averaging feat for trajs of the same id
featureVectors_tmp      = double(cell2mat({target_trajs.feature}'));
uniqueLabels = unique(target_labels);
for k = 1:length(uniqueLabels)
   l = uniqueLabels(k);
   inds = find(target_labels == l);
   meanVector = mean(featureVectors_tmp(inds,:),1);
   featureVectors_tmp(inds,:) = repmat(meanVector, length(inds),1);
end

source_feat_vector = {source_traj.feature};
target_feat_vectors = {target_trajs.feature};
for k = 1:length(target_feat_vectors)
   target_feat_vectors{k} = featureVectors_tmp(k,:); 
end

result.labels = target_labels;
result.observations = target_labels;
fprintf('merging trajectories in the only appearance group\n');
sameLabels  = pdist2(source_label, target_labels) == 0;

% compute appearance and spacetime scores
appearanceCorrelation = getAppearanceMatrix(source_feat_vector,target_feat_vectors, params.threshold,params.diff_p,params.diff_n,params.step);
[spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityL2_5(source_traj,target_trajs);
correlationMatrix = ...
        1 * appearanceCorrelation + ...
        params.alpha*(spacetimeAffinity).*(1-indifferenceMatrix);
    
    
correlationMatrix(impossibilityMatrix) = -Inf;

    
%     correlationMatrix(sameLabels) = 100;
    
% show appearance group tracklets
%     if VISUALIZE, trajectoriesVisualizePart2; end
    
% find the biggest match
[correlation,i]=max(correlationMatrix);
if correlation>threshold
    to_append_label = result.labels(i);
    i_s = find(result.labels==to_append_label);
    to_merge_trajs = target_trajs(i_s);
    % time overlapping check
    for k = 1:length(to_merge_trajs)
        if intersect(to_merge_trajs(k).data(:,9),source_traj.data(:,9))
            fprintf( 'Found duplicate ID/Frame pairs, try to remedy');
        
        end
    end
    
        
    result.labels(i)=0;
end
    
[~,id]              = sort(result.observations);
result.observations = result.observations(id);
result.labels       = result.labels(id);

end


