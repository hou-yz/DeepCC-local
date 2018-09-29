function [ appearanceMatrix ] = getAppearanceMatrix(featureVectors, threshold, half_dist )

% Computes the appearance affinity matrix

features = double(cell2mat(featureVectors'));
dist = pdist2(features, features);
if half_dist
    half_dist=threshold;
end
appearanceMatrix = (threshold - dist)/ half_dist;


