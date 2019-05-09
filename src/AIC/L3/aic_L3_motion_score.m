function [st_affinity,impossibility] = aic_L3_motion_score(opts,trajectories)
%AIC_L3_MOTION_SCORE Summary of this function goes here
%   Detailed explanation goes here
st_affinity = zeros(length(trajectories));
impossibility   = zeros(length(trajectories));
start_le_end = logical(triu(ones(length(trajectories)),1));

intervals    = zeros(length(trajectories),2);
start_points = zeros(length(trajectories),2);
end_points   = zeros(length(trajectories),2);

start_speeds = zeros(length(trajectories),2);
end_speeds   = zeros(length(trajectories),2);
for i = 1:length(trajectories)
    intervals(i,:)    = [trajectories(i).data(1,9),trajectories(i).data(end,9)];
    start_points(i,:) = trajectories(i).data(1,7:8);
    end_points(i,:)   = trajectories(i).data(end,7:8);
    frames            = trajectories(i).data(:,9);
    traj_length       = length(frames);
    start_indices     = 1:min(1+4,traj_length);
    end_indices       = max(1,traj_length-4):traj_length;
    start_speeds(i,:) = trajectories(i).data(start_indices(end),7:8) - trajectories(i).data(start_indices(1),7:8);
    
    end_speeds(i,:)   = trajectories(i).data(end_indices(end),7:8) - trajectories(i).data(end_indices(1),7:8);
    
end
iCams = [trajectories.camera];

long_time       = intervals(:,2)' - intervals(:,1);
consider = long_time>0; consider_nt = ~consider'; % not & tracpose
long_distance_x = end_points(:,1)' - start_points(:,1);
long_distance_y = end_points(:,2)' - start_points(:,2);
long_distance = sqrt(long_distance_x.^2 + long_distance_y.^2);
needed_v   = long_distance./(long_time+10^-12)*10;
needed_v_x = long_distance_x./(long_time+10^-12)*10;
needed_v_y = long_distance_y./(long_time+10^-12)*10;
needed_v_x(~consider) = 0; needed_v_y(~consider) = 0;
long_time(~consider) = 0;  long_distance(~consider) = 0;
% impossibility((long_distance - 10)./(long_time+10^-12) * 10 > opts.identities.speed_limit(1)) = 1;
% impossibility((long_distance + 10)./(long_time+10^-12) * 10 < opts.identities.speed_limit(2)) = 1;



traj_velocities = (end_points-start_points)./(intervals(:,2) - intervals(:,1))*10;
v_dist_euc = pdist2(traj_velocities,traj_velocities,'euclidean');
v_dist_cos = pdist2(traj_velocities,traj_velocities,'cosine');
needed_v_dist_euc = zeros(length(trajectories));
needed_v_dist_cos = zeros(length(trajectories));
for i = 1:length(trajectories)
    needed_v_i = [needed_v_x(i,:);needed_v_y(i,:)]';
    needed_v_dist_euc(i,:) = pdist2(traj_velocities(i,:),needed_v_i,'euclidean');
    needed_v_dist_cos(i,:) = pdist2(traj_velocities(i,:),needed_v_i,'cosine');    
end
impossibility(v_dist_euc > opts.identities.speed_limit(1)) = 1;
violators = (v_dist_cos > pi/2) | (needed_v_dist_cos > pi/2);  %
% violators(logical(ismember(iCams,opts.identities.allow_acute_cams) .* ismember(iCams,opts.identities.allow_acute_cams)')) = 0;
impossibility(violators) = 1;

impossibility(~start_le_end) = 0;
impossibility = impossibility + impossibility';

impossibility(iCams == iCams')  = 1;
impossibility = logical(impossibility);
end

