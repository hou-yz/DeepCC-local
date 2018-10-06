function [outputIdentities,if_updated,repairedIdentities,skip_ind] = linkL3Identities( opts, inputIdentities, sourceIdentityInd )

sourceIdentity = inputIdentities(sourceIdentityInd);
sourceTraj = sourceIdentity.trajectories(end);
target_iCams = 1:8;
target_iCams = target_iCams(opts.identities.consecutive_icam_matrix(sourceTraj.camera,:)==1);
target_reintro_time = opts.identities.reintro_time(sourceTraj.camera);

% find current, old, and future tracklets
targetIdentitiesInd    = findOneHopTrajectoriesInWindow(inputIdentities,target_iCams, sourceTraj.startFrame, sourceTraj.startFrame+target_reintro_time);
targetIdentities       = inputIdentities(targetIdentitiesInd);


repairedIdentities = [];
skip_ind=0;
% safety check
if length(targetIdentities) <= 1
    outputIdentities = inputIdentities;
    if_updated = 0;
    return;
end


% select tracklets that will be selected in association. For previously
% merged identities we select only the last three trajectoris.
inAssociation = []; target_id_trajs = []; target_id_labels = [];
for i = 1 : length(targetIdentities)
   for k = 1 : length(targetIdentities(i).trajectories) 
       target_id_trajs    = [target_id_trajs; targetIdentities(i).trajectories(k)]; %#ok
       target_id_labels   = [target_id_labels; i]; %#ok
       
       inAssociation(length(target_id_labels)) = false; %#ok
       % only consider the last traj
       if k == length(targetIdentities(i).trajectories)
           inAssociation(length(target_id_labels)) = true; %#ok
       end
       
   end
end
inAssociation = logical(inAssociation);

% show all tracklets
% if VISUALIZE, trajectoriesVisualizePart1; end

% solve the graph partitioning problem for each appearance group
[result,correlation] = solveL3Identities(opts,sourceTraj,sourceIdentityInd, target_id_trajs(inAssociation), target_id_labels(inAssociation));

% merge back solution. Tracklets that were associated are now merged back
% with the rest of the tracklets that were sharing the same trajectory
labels = target_id_labels; 
labels(inAssociation) = result.labels;
count = 1;
for i = 1 : length(inAssociation)
   if inAssociation(i) > 0
      labels(target_id_labels == target_id_labels(i)) = result.labels(count);
      count = count + 1;
   end
end

% Merge

    inds = find(labels == 0);
    if_updated = numel(inds);
    mergedIdentities = sourceIdentity;
    for k = 1:numel(inds)
        mergedIdentities.trajectories(end+1) = target_id_trajs(inds(k));
        mergedIdentities.startFrame = min(mergedIdentities.startFrame, mergedIdentities.trajectories(end).startFrame);
        mergedIdentities.endFrame   = max(mergedIdentities.endFrame, mergedIdentities.trajectories(end).endFrame);
        mergedIdentities.iCams = [mergedIdentities.iCams,mergedIdentities.trajectories(end).camera];
    
        mergedIdentities.trajectories = sortStruct(mergedIdentities.trajectories,'startFrame');
    end
    if if_updated
        if opts.visualize
            figure(1)
            for k = 1:length(mergedIdentities.trajectories)
                img = mergedIdentities.trajectories(k).snapshot;
                subplot(1,length(mergedIdentities.trajectories),k)
                imshow(img)
                if mergedIdentities.trajectories(k).startFrame == sourceIdentity.trajectories(end).startFrame
                    title("source ID: "+num2str(sourceIdentityInd)+newline+"correlation: "+num2str(correlation))
                end
            end
%             if if_updated>1
%                 figure(2)
%                 for k = 1:length(inds)
%                     img = target_id_trajs(inds(k)).snapshot;
%                     subplot(1,length(inds),k)
%                     imshow(img)
%                 end
%             end
        end
    end
 

% merge co-identified trajectories
outputIdentities = inputIdentities;
assert(length(unique(target_id_labels(inds)))<=1, "removed more than one id in one iter")
outputIdentities(targetIdentitiesInd(unique(target_id_labels(inds)))) = [];
outputIdentities(sourceIdentityInd) = mergedIdentities';


    % time overlapping check
    data = [];
    for k = 1:length(mergedIdentities.trajectories)
        data = [data; mergedIdentities.trajectories(k).data];
    end
    frames = unique(data(:,9));
    if length(frames) ~= size(data,1)
        fprintf( 'Found duplicate ID/Frame pairs, restore to source');
        remedy_labels = 1:length(mergedIdentities.trajectories);
        repairedIdentities=remedyL3Identities(opts,mergedIdentities.trajectories,remedy_labels);
        
        repeated_id = zeros(1,length(repairedIdentities));
        for k = 1:length(repairedIdentities)
            if repairedIdentities(k).startFrame==outputIdentities(sourceIdentityInd-1).startFrame && repairedIdentities(k).endFrame==outputIdentities(sourceIdentityInd-1).endFrame
                repeated_id(k) = 1;
            end
        end
        if sum(repeated_id)
            outputIdentities=[outputIdentities(1:sourceIdentityInd-1),repairedIdentities(~repeated_id),outputIdentities(sourceIdentityInd+1:end)];
            skip_ind = max(find(repeated_id))-1;
        else
            outputIdentities=[outputIdentities(1:sourceIdentityInd-1),repairedIdentities,outputIdentities(sourceIdentityInd+1:end)];
        end
        if_updated=length(repairedIdentities(1).trajectories)-1;
        if opts.visualize
                figure(2)
                for k = 1:length(repairedIdentities(1).trajectories)
                    img = repairedIdentities(1).trajectories(k).snapshot;
                    subplot(1,length(repairedIdentities(1).trajectories),k)
                    imshow(img)
                end
        end
    end
    
end

