function [ correlation ] = getAppearanceSubMatrix(observations, featureVectors, threshold, half_distance )

features = cell2mat(featureVectors.appearance(observations));
dist = pdist2(features, features);
if half_distance==0:
    half_distance=threshold;
correlation = (threshold - dist)/ half_distance;




