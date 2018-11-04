function result = solveInGroups(opts, tracklets, labels, iCam,hyper_score_param)

global trajectorySolverTime;

params = opts.trajectories;
if length(tracklets) < params.appearance_groups
    params.appearanceGroups = 1;
end

% set threshold accordingly if trained on separate icams
if length(params.threshold)==8
    threshold = params.threshold(iCam);
else
    threshold = params.threshold;
end

if length(params.diff_p)==8
    diff_p = params.diff_p(iCam);
    diff_n = params.diff_n(iCam);
else
    diff_p = params.diff_p;
    diff_n = params.diff_n;
end

if isempty(tracklets)
    result.labels       = [];
    result.observations = [];
    return
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
    [spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinity(tracklets(indices), params.beta, params.speed_limit, params.indifference_time);
    if params.compute_score
        % compute appearance and spacetime scores
        appearanceAffinity = getAppearanceMatrix(featureVectors(indices),featureVectors(indices), threshold, diff_p,diff_n,params.step);
        % compute the correlation matrix
        correlationMatrix = appearanceAffinity + params.alpha*(spacetimeAffinity-1);
        correlationMatrix = correlationMatrix .* indifferenceMatrix;
    else
        correlationMatrix = getHyperScore(featureVectors(indices),tracklets(indices),hyper_score_param,params.alpha);
    end
    
    
    correlationMatrix(impossibilityMatrix == 1) = -inf;
    correlationMatrix(sameLabels) = 1;
    
    % show appearance group tracklets
%     if opts.visualize, trajectoriesVisualizePart2; end
    
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


