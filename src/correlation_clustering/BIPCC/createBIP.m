function [ f, A, b ] = createBIP( W )
% Creates a Binary Integer Program for a given correlation matrix
%
% Input
%   W -  correlation matrix
%
% Output
%   f - objective function
%   A, b - constraints of type Ax <= b
%
% Ergys Ristani
% Duke University

fprintf('Num vars: %d\n\n', size(W,1));

constraintsTime = tic;

N = size(W,1);

% f is the objective function f = sum{W} w_uv * x_uv 
f = zeros(N*(N-1)/2,1);

% pairs - a list of all edges (u,v), v>u
pairs = zeros(N,N-1);

pos = 1;
for i = 1:N-1
    for j = i + 1:N
        pairs(i, j) = pos; 
        f(pos) = W(i,j);
        pos = pos + 1;
    end
end

% Constraint matrix  x_uv + x_ut - x_vt <= 1 represented by
% Ax<=b

% Computie all possible triples between observations

if N>=3
   
    triples = combinator(N,3,'c','no'); % Equivalent to combnk(1:N,3) but faster

    % Translate triples to observation unique IDs
    % Observation (i,j) has pairs(i,j) as unique ID

    idx = zeros(size(triples));

    idx(:,1) = pairs(sub2ind(size(pairs), triples(:,1),triples(:,2))); 
    idx(:,2) = pairs(sub2ind(size(pairs), triples(:,1),triples(:,3)));
    idx(:,3) = pairs(sub2ind(size(pairs), triples(:,2),triples(:,3)));

    % Express x_uv + x_ut - x_vt <= 1 as three constraints by permutation
    idx_x3 = kron(idx,[1 1 1]');
    permutations = idx_x3;
    permutations(2:3:end,:)=circshift(idx_x3(2:3:end,:)',1)';
    permutations(3:3:end,:)=circshift(idx_x3(3:3:end,:)',2)';

    idx = permutations';

    rows = kron(1:(size(triples,1)*3),[1,1,1])';
    cols = idx(:);
    values = kron(ones(size(triples,1)*3,1),[1 1 -1]');

    % A - constraints matrix
    A = sparse(rows,cols,values);
    b = ones(size(A,1),1);
    
    % All columns in A belonging to -inf assignments are removed for
    % memory efficiency. This affects how the partial solution is built
    % in correlationClustering.m
    infIndices = pairs(W==-inf);
    infIndices(infIndices==0) = [];
    
    A(:,infIndices) = []; 
    f(infIndices) = [];
    
elseif N == 2

    % No constraints
    A = sparse([0]);
    b = [0];

else 

    % N is 0 or 1
    A = sparse([0]);
    b = [0];
    f = W;
end

fprintf('Assembling BIP. '); toc(constraintsTime);
fprintf('Number of constraints: %d\n\n', size(A,1));



