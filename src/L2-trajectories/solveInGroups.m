function result = solveInGroups(opts, tracklets, labels)

global trajectorySolverTime;

params = opts.trajectories;
if length(tracklets) < params.appearance_groups
    params.appearanceGroups = 1;
end

featureVectors      = {tracklets.feature};
% adaptive number of appearance groups
if params.appearance_groups == 0
    % Increase number of groups until no group is too large to solve 
    while true
        params.appearance_groups = params.appearance_groups + 1;
        appearanceGroups    = kmeans( cell2mat(featureVectors'), params.appearance_groups, 'emptyaction', 'singleton', 'Replicates', 10);
        uid = unique(appearanceGroups);
        freq = [histc(appearanceGroups(:),uid)];
        largestGroupSize = max(freq);
        % The BIP solver might run out of memory for large graphs
        if largestGroupSize <= 150
            break
        end
    end
else
    % fixed number of appearance groups
    appearanceGroups    = kmeans( cell2mat(featureVectors'), params.appearance_groups, 'emptyaction', 'singleton', 'Replicates', 10);
end
% solve separately for each appearance group
allGroups = unique(appearanceGroups);

result_appearance = cell(1, length(allGroups));
for i = 1 : length(allGroups)
    
    fprintf('merging tracklets in appearance group %d\n',i);
    group       = allGroups(i);
    indices     = find(appearanceGroups == group);
    sameLabels  = pdist2(labels(indices), labels(indices)) == 0;
    
    % compute appearance and spacetime scores
    appearanceAffinity = getAppearanceMatrix(featureVectors(indices), params.threshold);
    [spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinity(tracklets(indices), params.beta, params.speed_limit, params.indifference_time);
    
    % compute the correlation matrix
    correlationMatrix = appearanceAffinity + spacetimeAffinity - 1;
    correlationMatrix = correlationMatrix .* indifferenceMatrix;
    
    correlationMatrix(impossibilityMatrix == 1) = -inf;
    correlationMatrix(sameLabels) = 1;
    
    % show appearance group tracklets
    if opts.visualize, trajectoriesVisualizePart2; end
    
    % solve the optimization problem
    solutionTime = tic;
    if strcmp(opts.optimization,'AL-ICM')
        result_appearance{i}.labels  = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        result_appearance{i}.labels  = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        result_appearance{i}.labels  = BIPCC(correlationMatrix, initialSolution);
    end
    
    trajectorySolutionTime = toc(solutionTime);
    trajectorySolverTime = trajectorySolverTime + trajectorySolutionTime;
    
    result_appearance{i}.observations = indices;
end


% collect independent solutions from each appearance group
result.labels       = [];
result.observations = [];

for i = 1:numel(unique(appearanceGroups))
    result = mergeResults(result, result_appearance{i});
end

[~,id]              = sort(result.observations);
result.observations = result.observations(id);
result.labels       = result.labels(id);


