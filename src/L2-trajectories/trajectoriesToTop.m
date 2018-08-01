function data = trajectoriesToTop( trajectories )
%TRAJECTORIESTOTOP Summary of this function goes here
%   Detailed explanation goes here

data = zeros(0,8);

for i = 1:length(trajectories)
    
    traj = trajectories(i);
    
    for k = 1:length(traj.tracklets)
       
        newdata = traj.tracklets(k).data;
        newdata(:,2) = i;
        data = [data; newdata];
        
        
    end

end

