function [ appearanceMatrix ] = getAppearanceMatrix(featureVectors, threshold, diff_p,diff_n ,step)

% Computes the appearance affinity matrix

features = double(cell2mat(featureVectors'));
dist = pdist2(features, features);
if diff_p==0
    diff_p=threshold;
    diff_n=threshold;
end
correlation = (threshold - dist);
correlation_p = correlation/diff_p;
correlation_n = correlation/diff_n;
if step
    correlation(correlation>0) = 1;
    correlation(correlation<0) = -1;
    correlation(correlation==0) = 0;
else
correlation(correlation>=0) = correlation_p(correlation_p>=0);
correlation(correlation<0)  = correlation_n(correlation_n<0);
end
appearanceMatrix = correlation;
end
