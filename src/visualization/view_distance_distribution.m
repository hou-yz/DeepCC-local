function view_distance_distribution(opts)
% Shows the distance distribution between positive and negative pairs for
% the appearance model specified in opts.net_weights

data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',',');

labels = data.Var1;
paths  = data.Var2;
%% Compute features
features = h5read(fullfile('src/triplet-reid',  opts.net.experiment_root, 'duke_test_embeddings.h5'),'/emb');
features = features';
dist = pdist2(features,features);
%% Visualize distance distribution
same_label = triu(pdist2(labels,labels) == 0,1);
different_label = triu(pdist2(labels,labels) ~= 0);
pos_dists = dist(same_label);
neg_dists = dist(different_label);

figure;
histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
hold on;
histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
title('Normalized distribution of distances among positive and negative pairs');
legend('Positive','Negative');

% figure;
% histogram(pos_dists,100);
% hold on;
% histogram(neg_dists,100);
% title('Unnormalized distribution of distances among positive and negative pairs');
% legend('Positive','Negative');

