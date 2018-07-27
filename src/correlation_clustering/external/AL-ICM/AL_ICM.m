function l = AL_ICM(w, ig)
%
% adaptive-label ICM
%
% Usage:
%   l = AL_ICM(w, [ig])
%
% Inputs:
%   w - sparse, symmetric affinity matrix containng BOTH positive and
%       negative entries (attraction and repultion).
%   ig- initial guess labeling (optional)
%
% Output:
%   l - labeling of the nodes into different clusters.
%



n = size(w,1);

LIMIT = 100*ceil(reallog(n));

% ignore diagonal
w = w - spdiags( spdiags(w,0), 0, n, n);

% initial guess
if nargin == 1 || isempty(ig) || numel(ig) ~= n
    l = ones(n, 1);
else
    [~, ~, l] = unique(ig);    
end

l = 1:n;

for itr=1:LIMIT
    nl = pure_potts_icm_iter_mex(w, l);
    if isequal(l,nl)
        break;
    end
    l = nl;
    if mod(itr, 15) == 0 % removing empty labels make code run faster!
        [~, ~, l] = unique(l);
        l = pure_potts_icm_iter_mex(w, l);
    end
end

% if itr == LIMIT
%     warning('ML_ICM:itr','reached LIMIT');
% end

% remove "empty" labels
[~, ~, l] = unique(l);

