function result = solveInGroupsIdentities(opts, trajectories, labels, VISUALIZE)

global identitySolverTime;

params = opts.identities;
if length(trajectories) < params.appearance_groups
    params.appearanceGroups = 1;
end

% adaptive number of appearance groups
if params.appearance_groups == 0
    params.appearance_groups = 1 + floor(length(trajectories)/120);
end

% fixed number of appearance groups
featureVectors_tmp      = double(cell2mat({trajectories.feature}'));
uniqueLabels = unique(labels);
for k = 1:length(uniqueLabels)
   label = uniqueLabels(k);
   inds = find(labels == label);
   meanVector = mean(featureVectors_tmp(inds,:),1);
   featureVectors_tmp(inds,:) = repmat(meanVector, length(inds),1);
end

featureVectors      = {trajectories.feature};

for k = 1:length(featureVectors)
   featureVectors{k} = featureVectors_tmp(k,:); 
end

appearanceGroups    = kmeans( cell2mat(featureVectors'), params.appearance_groups, 'emptyaction', 'singleton', 'Replicates', 10);

% solve separately for each appearance group
allGroups = unique(appearanceGroups);

% figure; 
% for k = 1:length(trajectories)
%     subplot(1,length(trajectories),k); 
%     imshow(trajectories(k).snapshot); 
% end


group_results = cell(1, length(allGroups));
for i = 1 : length(allGroups)
    
    fprintf('merging trajectories in appearance group %d\n',i);
    group       = allGroups(i);
    indices     = find(appearanceGroups == group);
    sameLabels  = pdist2(labels(indices), labels(indices)) == 0;
    
    % compute appearance and spacetime scores
    appearanceCorrelation = getAppearanceMatrix(featureVectors(indices), params.threshold);
    [spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityL3(trajectories(indices));
    correlationMatrix = ...
        1 * appearanceCorrelation + ...
        5 * (spacetimeAffinity).*(1-indifferenceMatrix)-0.1;
    correlationMatrix = ...
        1 * appearanceCorrelation + ...
        (spacetimeAffinity).*(1-indifferenceMatrix);
    
    
    correlationMatrix(impossibilityMatrix) = -Inf;

    
    correlationMatrix(sameLabels) = 100;
    
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
    
    identitySolutionTime = toc(solutionTime);
    identitySolverTime = identitySolverTime + identitySolutionTime;
    
    group_results{i}.observations = indices;
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


