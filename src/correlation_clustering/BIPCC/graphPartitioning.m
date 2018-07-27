function [ X , objval ] = graphPartitioning( f, A, b, correlationMatrix, initialGuess )
% Finds the optimal partitioning of a correlation matrix
%
% Input:   
%   f - the objective function to maximize
%       sum w_uv * x_uv 
%   A, b - constraints of type Ax <= b
%   correlationMatrix
%   initialGuess (optinal)
%
% Output:
%   X - solution in the form of an upper triangular matrix in vector form
%   objval - value of the objective
%
% Ergys Ristani
% Duke University

partialSolution = f;
partialSolution(f == -inf) = 0;
partialSolution(f ~= -inf) = NaN;

if ~isempty(initialGuess)
	N = length(initialGuess);
	pos = 1;
	for i = 1:N-1
	    for j = i + 1:N
            if correlationMatrix(i,j) == -inf
                continue;
            end
            partialSolution(pos) = initialGuess(i) == initialGuess(j);
            pos = pos + 1;
	    end
	end
end

% Gurobi solver parameters
clear model;
model.obj = f;
model.A = sparse(A);
model.sense = '<';
model.rhs = b;
model.vtype = 'B'; % binary
model.modelsense = 'max';
model.start = partialSolution;
model.norelheuristic = 1;
clear params;

params = struct();
params.Presolve = 0;
% params.CliqueCuts = 2;
%params.outputflag = 0; % Silence gurobi
% params.NodefileStart = 0.1;
% params.TimeLimit = 10;

result = gurobi(model, params);
objval = result.objval;
X = [result.x];




