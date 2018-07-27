function [ labels ] = labelsFromAdjacency( X, W )
% Transforms the Binary Integer Program solution vector to a label vector
%
% Input
%   X - solution vector representing an upper triangular matrix
%   W - correlation matrix
%
% Output
%   labels - label vector
%
% Ergys Ristani
% Duke University

labels = [1:size(W,1)]';
pos = 1;

for i = 1:size(W,1)
   
    for j = i + 1:size(W,1)
        
        % Skip the -inf assignments that were previously removed from the
        % Binary Integer Program (createILP.m) for memory efficiency
        if W(i,j)== -inf
            continue;
        end
       
        if X(pos)==1
            labels(j) = labels(i);
        end
        
        pos = pos + 1;
    end
end

% Assign labels from 1 to n
ids = unique(labels);
tmplabels = labels;

for i = 1:length(ids)
    labels(tmplabels==ids(i)) = i;
end



