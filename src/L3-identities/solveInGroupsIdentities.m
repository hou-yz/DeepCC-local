function result = solveInGroupsIdentities(opts, trajectories, labels,appear_model_param,motion_model_param)

global identitySolverTime;

params = opts.identities;
if length(trajectories) < params.appearance_groups
    params.appearanceGroups = 1;
end

% adaptive number of appearance groups
% experience setting: one appear group every 120 traj's
if params.appearance_groups == 0
    params.appearance_groups = 1 + floor(length(trajectories)/120);
end

featureVectors      = {trajectories.feature};

% % set unified feat for trajs in same id
% featureVectors_tmp      = double(cell2mat({trajectories.feature}'));
% uniqueLabels = unique(labels);
% for k = 1:length(uniqueLabels)
%    label = uniqueLabels(k);
%    inds = find(labels == label);
%    meanVector = mean(featureVectors_tmp(inds,:),1);
%    featureVectors_tmp(inds,:) = repmat(meanVector, length(inds),1);
% end
% for k = 1:length(featureVectors)
%    featureVectors{k} = featureVectors_tmp(k,:); 
% end

appearanceGroups    = kmeans( cell2mat(featureVectors'), params.appearance_groups, 'emptyaction', 'singleton', 'Replicates', 10);

% solve separately for each appearance group
allGroups = unique(appearanceGroups);

% figure; 
% for k = 1:length(trajectories)
%     subplot(1,length(trajectories),k); 
%     imshow(trajectories(k).snapshot); 
% end

group_results = cell(1, length(allGroups));
same_id_correlation_matrix = [];
for i = 1 : length(allGroups)
    
    fprintf('merging trajectories in appearance group %d\n',i);
    group       = allGroups(i);
    indices     = find(appearanceGroups == group);
    sameLabels  = pdist2(labels(indices), labels(indices)) == 0;
    
    % compute appearance and spacetime scores
    if params.og_appear_score
        appearanceCorrelation = getAppearanceMatrix(featureVectors(indices),featureVectors(indices), params.threshold,params.diff_p,params.diff_n,params.step);
    else
        appearanceCorrelation = getHyperScore(featureVectors(indices),appear_model_param,opts.soft, params.threshold,params.diff_p,0);
    end
    if opts.dataset == 0
    	[spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityID(opts,trajectories(indices),opts.identities.consecutive_icam_matrix,opts.identities.reintro_time_matrix,opts.identities.optimal_filter);
    elseif opts.dataset == 1 || opts.dataset == 2
        impossibilityMatrix = zeros(length(trajectories(indices)));
        spacetimeAffinity = 0;
        indifferenceMatrix = 1;
        iCams = [trajectories(indices).camera];
        [~, impossibilityMatrix] = aic_L3_motion_score(opts,trajectories(indices));
%         smoothnessLoss = aic_SmoothnessMatrix(trajectories(indices), params.smoothness_interval_length);
%         impossibilityMatrix(smoothnessLoss>10) = 1;
        impossibilityMatrix(iCams == iCams')  = 1;
        impossibilityMatrix = logical(impossibilityMatrix);
    end
    
    if params.alpha
        correlationMatrix = 1 * appearanceCorrelation + params.alpha*(spacetimeAffinity).*(1-indifferenceMatrix);
        correlationMatrix(impossibilityMatrix) = -Inf;
    else
        correlationMatrix = appearanceCorrelation;
        correlationMatrix(impossibilityMatrix) = -Inf;
    end
    
    correlationMatrix(sameLabels) = max(10, correlationMatrix(sameLabels));
    
    % show appearance group tracklets
%     if VISUALIZE, trajectoriesVisualizePart2; end
    
    % solve the optimization problem
    solutionTime = tic;
    
    if strcmp(opts.optimization,'AL-ICM')
        group_results{i}.labels  = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        group_results{i}.labels  = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        group_results{i}.labels  = BIPCC(correlationMatrix,initialSolution);
    end
    
    unique_indices = unique(group_results{i}.labels);
    for j = 1:length(unique_indices)
        ind = find(group_results{i}.labels==unique_indices(j));
        for k = 1:length(ind)-1
            x=ind(k);
            y=ind(k+1);
            same_id_correlation_matrix=[same_id_correlation_matrix;correlationMatrix(x,y)];
        end
    end
    
    identitySolutionTime = toc(solutionTime);
    identitySolverTime = identitySolverTime + identitySolutionTime;
    
    group_results{i}.observations = indices;
    
%     if opts.visualize,view_tsne(correlationMatrix,group_results{i}.observations);end
end


% collect independent solutions from each appearance group
result.labels       = [];
result.observations = [];

for i = 1:numel(unique(appearanceGroups))
    result = mergeResults(result, group_results{i});
end

[~,id]              = sort(result.observations);
result.observations = result.observations(id);
result.labels       = result.labels(id);

end


