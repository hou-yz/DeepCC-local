function labels = KernighanLin(correlationMatrix)

if size(correlationMatrix,1) == 1
    labels = [1];
    return;
end
upper_tri = triu(ones(size(correlationMatrix)));
[sourceNode, targetNode] = find(upper_tri);
values = correlationMatrix(sub2ind(size(correlationMatrix),sourceNode,targetNode));
infidx = find(values == -Inf);

% [sourceNode,targetNode,values] = find(problemGraph.sparse_correlationMat_Upper);
rmIdx = bsxfun(@eq,sourceNode,targetNode);
% rmIdx(infidx) = 1;
sourceNode(rmIdx) = [];
targetNode(rmIdx) = [];
values(rmIdx) = [];
sourceNode = sourceNode -1;
targetNode = targetNode-1;

nNodes = numel(diag(upper_tri));
weightedGraph = [sourceNode,targetNode,values];

A = Multicut_KL_MEX(nNodes,size(weightedGraph,1),weightedGraph);
%output is of the form ((u1,v1,u2,v2,...) for all edges {u_i,v_i}
%index starting with 0.
% Converting back to matlab format
A_r = A(1:2:end)';
A_r(:,2) = A(2:2:end);
A_r= A_r+ 1; 
result = A_r;


G = graph;
G = addnode(G,nNodes);
G = addedge(G,result(:,1)',result(:,2)');
labels = conncomp(G)';



end