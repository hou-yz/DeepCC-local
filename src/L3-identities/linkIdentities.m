function outputIdentities = linkIdentities( opts, inputIdentities, startTime, endTime)


% find current, old, and future tracklets
currentIdentitiesInd    = findTrajectoriesInWindow(inputIdentities, startTime, endTime);
currentIdentities       = inputIdentities(currentIdentitiesInd);

% safety check
if length(currentIdentities) <= 1
    outputIdentities = inputIdentities;
    return;
end

% select tracklets that will be selected in association. For previously
% computed trajectories we select only the last three tracklets.
inAssociation = []; trajectories = []; trajectoryLabels = [];
for i = 1 : length(currentIdentities)
   for k = 1 : length(currentIdentities(i).trajectories) 
       trajectories        = [trajectories; currentIdentities(i).trajectories(k)]; %#ok
       trajectoryLabels   = [trajectoryLabels; i]; %#ok
       
       inAssociation(length(trajectoryLabels)) = false; %#ok
       if k >= length(currentIdentities(i).trajectories) - 5
           inAssociation(length(trajectoryLabels)) = true; %#ok
       end
       
   end
end
inAssociation = logical(inAssociation);

% show all tracklets
% if VISUALIZE, trajectoriesVisualizePart1; end

% solve the graph partitioning problem for each appearance group
result = solveInGroupsIdentities(opts, trajectories(inAssociation), trajectoryLabels(inAssociation));

% merge back solution. Tracklets that were associated are now merged back
% with the rest of the tracklets that were sharing the same trajectory
labels = trajectoryLabels; 
labels(inAssociation) = result.labels;
count = 1;
for i = 1 : length(inAssociation)
   if inAssociation(i) > 0
      labels(trajectoryLabels == trajectoryLabels(i)) = result.labels(count);
      count = count + 1;
   end
end

% Merge
mergedIdentities = [];
uniqueLabels = unique(labels);
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);
    inds = find(labels == label);
    identity = [];
    identity.startFrame = Inf;
    identity.endFrame = -Inf;
    for k = 1:length(inds)
        identity.trajectories(k) = trajectories(inds(k));
        identity.startFrame = min(identity.startFrame, identity.trajectories(k).startFrame);
        identity.endFrame   = max(identity.endFrame, identity.trajectories(k).endFrame);
    end
    identity.trajectories = sortStruct(identity.trajectories,'startFrame');

    
    mergedIdentities = [mergedIdentities; identity];
    
    data = [];
    for k = 1:length(identity.trajectories)
        data = [data; identity.trajectories(k).data];
    end
    frames = unique(data(:,9));
    assert(length(frames) == size(data,1), 'Found duplicate ID/Frame pairs');
    
end

% merge co-identified trajectories
outputIdentities = inputIdentities;
outputIdentities(currentIdentitiesInd) = [];
outputIdentities = [mergedIdentities', outputIdentities];

