function trajectoriesInWindow = findOneHopTrajectoriesInWindow(targetIdentities,target_iCams, startTime, endTime)
trajectoriesInWindow = [];

if isempty(targetIdentities), return; end

identitiesLengths = zeros(1,length(targetIdentities));
for i = 1:length(targetIdentities)
    identitiesLengths(i) = length(targetIdentities(i).trajectories);
end
    
trajectoriesCamera = [targetIdentities.iCams];
traj_in_target_icams = ismember(trajectoriesCamera,target_iCams);
identities_in_target_icams = zeros(1,length(targetIdentities));
k=1;
for i = 1:length(targetIdentities)
    identities_in_target_icams(i)=sum(traj_in_target_icams(k:k+identitiesLengths(i)-1));
    k = k+identitiesLengths(i);
end

identities_in_target_range = ([targetIdentities.endFrame] >= startTime) .* ([targetIdentities.startFrame] <= endTime);

trajectoriesInWindow  = find(identities_in_target_icams .* identities_in_target_range );

end
