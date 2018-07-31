function [ appearanceMatrix ] = getAppearanceMatrix(featureVectors, threshold )

% Computes the appearance affinity matrix

features = double(cell2mat(featureVectors'));
dist = pdist2(features, features);
appearanceMatrix = (threshold - dist)/ threshold;


