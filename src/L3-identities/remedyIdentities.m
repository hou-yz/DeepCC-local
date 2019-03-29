function [updatedIdentities] = remedyIdentities(opts,trajs,labels)
params = opts.identities;

feat_vectors = {trajs.feature};

result.labels = labels;
result.observations = labels;
fprintf('re-merging trajectories\n');
sameLabels  = pdist2(labels, labels) == 0;

% compute appearance and spacetime scores
appearanceCorrelation = getAppearanceMatrix(feat_vectors,feat_vectors, params.threshold,params.diff_p,params.diff_n,params.step);
[spacetimeAffinity, impossibilityMatrix, indifferenceMatrix] = getSpaceTimeAffinityID(trajs,opts.identities.consecutive_icam_matrix,opts.identities.reintro_time_matrix,opts.identities.optimal_filter);
    if params.alpha
        correlationMatrix = 1 * appearanceCorrelation + params.alpha*(spacetimeAffinity).*(1-indifferenceMatrix);
        correlationMatrix(impossibilityMatrix) = -Inf;
    else
        correlationMatrix = appearanceCorrelation;
    end

    
%     correlationMatrix(sameLabels) = 100;

    if strcmp(opts.optimization,'AL-ICM')
        result.labels  = AL_ICM(sparse(correlationMatrix));
    elseif strcmp(opts.optimization,'KL')
        result.labels  = KernighanLin(correlationMatrix);
    elseif strcmp(opts.optimization,'BIPCC')
        initialSolution = KernighanLin(correlationMatrix);
        result.labels  = BIPCC(correlationMatrix,initialSolution);
    end    

% show appearance group tracklets
%     if VISUALIZE, trajectoriesVisualizePart2; end

    
[~,id]              = sort(result.observations);
result.observations = result.observations(id);
result.labels       = result.labels(id);

%% merge back identities
updatedIdentities = [];
uniqueLabels = unique(result.labels);
for i = 1:length(uniqueLabels)
    l = uniqueLabels(i);
    inds = find(result.labels == l);
    identity = [];
    identity.startFrame = Inf;
    identity.endFrame = -Inf;
    identity.iCams = [];
    for k = 1:length(inds)
        identity.trajectories(k) = trajs(inds(k));
        identity.startFrame = min(identity.startFrame, identity.trajectories(k).startFrame);
        identity.endFrame   = max(identity.endFrame, identity.trajectories(k).endFrame);
        identity.iCams = [identity.iCams,identity.trajectories(end).camera];
    end
    identity.trajectories = sortStruct(identity.trajectories,'startFrame');
    
    updatedIdentities = [updatedIdentities, identity];
    
    data = [];
    for k = 1:length(identity.trajectories)
        data = [data; identity.trajectories(k).data];
    end
    frames = unique(data(:,9));
    
%     assert(length(frames) == size(data,1), 'Found duplicate ID/Frame pairs');
    
end

end


