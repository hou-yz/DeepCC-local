function data = identities2mat( identities )

data = zeros(0,8);

for i = 1:length(identities)
    
    identity = identities(i);
    
    for k = 1:length(identity.trajectories)
       
        newdata = identity.trajectories(k).data;
        newdata(:,2) = i;
        cam_data = identity.trajectories(k).camera * ones(size(identity.trajectories(k).data,1),1) ;
        data = [data; cam_data, newdata];
        
        
    end

end

