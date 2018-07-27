function labels = BIPCC( correlationMatrix, initialGuess )
% Solves a graph partitioning problem as a Binary Integer Program
%
% Input
%   correlationMatrix
%   initialGuess - initial guess label vector (optional)
%
% Output
%   labels - optimal label vector
%
% Ergys Ristani
% Duke University

[f, A, b] = createBIP(correlationMatrix);

[X, ~] = graphPartitioning(f, A, b, correlationMatrix, initialGuess);

% Create a label vector from the BIP solution

labels = labelsFromAdjacency(X, correlationMatrix);


