function [ correlation ] = getAppearanceSubMatrix(observations, featureVectors, threshold )

features = cell2mat(featureVectors.appearance(observations));
dist = pdist2(features, features);
correlation = (threshold - dist)/ threshold;




