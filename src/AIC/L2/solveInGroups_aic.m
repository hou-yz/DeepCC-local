function result = solveInGroups_aic(opts, tracklets, labels, iCam,appear_model_param,motion_model_param)

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
% when aic, realdata == data 
realdataInTracklets = {tracklets.realdata};
dataInTracklets     = {tracklets.data};
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
    
    if params.og_appear_score
        appearanceAffinity = getAppearanceMatrix(featureVectors(indices),featureVectors(indices), threshold, diff_p,diff_n,params.step);
    else
        appearanceAffinity = getHyperScore(featureVectors(indices),appear_model_param,opts.soft,threshold, diff_p,0);
    end
    
    if opts.dataset == 0
        [spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinity(tracklets(indices), params.beta, params.speed_limit, params.indifference_time, iCam);
        spacetimeAffinity = spacetimeAffinity-1;
    elseif opts.dataset == 1 || opts.dataset == 2
%         [spacetimeAffinity, impossibilityMatrix] = aic_VelocityTimeMatrix(opts,dataInTracklets(indices), iCam, params.smoothness_interval_length);
        impossibilityMatrix = zeros(length(dataInTracklets(indices)));
        spacetimeAffinity   = zeros(length(dataInTracklets(indices)));
        
        smoothnessLoss = aic_SmoothnessMatrix(dataInTracklets(indices), params.smoothness_interval_length);
        if ~ismember(iCam,opts.trajectories.less_smooth_cams) 
            impossibilityMatrix(smoothnessLoss>2) = 1;
        else
            impossibilityMatrix(smoothnessLoss>4) = 1;
        end
        impossibilityMatrix = logical(impossibilityMatrix);
        
        
        appearanceAffinity = appearanceAffinity;
        indifferenceMatrix = 1;
%         indifferenceMatrix = min(1, -log(timeIntervalMatrix/params.window_width));
%         indifferenceMatrix = 1 - sigmf(timeIntervalMatrix,[0.1, params.indifference_time/2]);
%         impossibilityMatrix = false;%impossibility_frame_overlap([tracklets(indices).startFrame]',[tracklets(indices).endFrame]');    
    end
    % compute the correlation matrix
    if params.alpha
        correlationMatrix = appearanceAffinity + params.alpha*spacetimeAffinity;
        correlationMatrix = correlationMatrix .* indifferenceMatrix.^params.use_indiff;
        correlationMatrix(impossibilityMatrix == 1) = -inf;
    else
        correlationMatrix = appearanceAffinity;
        correlationMatrix(impossibilityMatrix == 1) = -inf;
    end
    
    if sum(sum(isnan(correlationMatrix)))
        correlationMatrix
    end
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
    elseif strcmp(opts.optimization,'hierachical')
        num_cluster = ceil(sum(indices>0)/2);
        result_appearance{i}.labels  = hierarchical_cluster(correlationMatrix,num_cluster);
    end
    
    trajectorySolutionTime = toc(solutionTime);
    trajectorySolverTime = trajectorySolverTime + trajectorySolutionTime;
    
    result_appearance{i}.observations = indices;
    identities = result_appearance{i}.labels;
    consider_tracklets = tracklets(indices);
    % Show clustered detections
    if opts.visualize, trajectoriesVisualizePart2_5; end
%     if opts.visualize,view_tsne(correlationMatrix,result_appearance{i}.labels);end
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
end

