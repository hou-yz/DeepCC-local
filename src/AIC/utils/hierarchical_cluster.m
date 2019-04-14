function labels = hierarchical_cluster(correlationMatrix,num_cluster)
%HIERACHICAL_CLUSTER Summary of this function goes here
%   Detailed explanation goes here
Z = linkage(correlationMatrix);
labels = cluster(Z,'maxclust',num_cluster);
end

