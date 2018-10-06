function [ appearanceMatrix ] = getAppearanceMatrix(source_feat_vectors,target_feat_vectors, threshold, diff_p,diff_n ,step)

% Computes the appearance affinity matrix

source_feats = double(cell2mat(source_feat_vectors'));
target_feats = double(cell2mat(target_feat_vectors'));
dist = pdist2(source_feats, target_feats);
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
