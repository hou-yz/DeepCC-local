opts = get_opts();
data = readtable('src/visualization/file_list.csv', 'Delimiter',','); % gt@1fps
% data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',','); % reid

opts.net.experiment_root = 'experiments/fc256_6fps_epoch45';
labels = data.Var1;
paths  = data.Var2;
%% Compute features
features = h5read(fullfile(opts.net.experiment_root, 'features.h5'),'/emb');
features = features';
% pooling
pooling = 2;
labels = labels(pooling:pooling:length(labels),:);
features = features(pooling:pooling:length(features),:);
dist = pdist2(features,features);
%% Visualize distance distribution
same_label = triu(pdist2(labels,labels) == 0,1);
different_label = triu(pdist2(labels,labels) ~= 0);
pos_dists = dist(same_label);
neg_dists = dist(different_label);
pos_99th = prctile(pos_dists,99);
neg_5th = prctile(neg_dists,5);
mid = mean([mean(pos_dists),mean(neg_dists)]);



fig = figure;
hold on;
h_P = histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
h_N = histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
title('Normalized distribution of distances among positive and negative pairs');
legend('Positive','Negative');
neg_str = ['\downarrow dist_P less than the 5th percentile dist_N: ',num2str(sum(pos_dists<neg_5th)/length(pos_dists))];
mid_str = ['\downarrow dist_P less than the mid value: ',num2str(sum(pos_dists<mid)/length(pos_dists))];
pos_str = "\downarrow dist_P 99th";
info_str = "dist_P 99th: "+num2str(pos_99th)+newline+"dist_N 5th: "+num2str(neg_5th)+newline+"mid value: "+num2str(mid);

text(neg_5th,0.025,neg_str)
text(mid,0.02,mid_str)
text(pos_99th,0.03,pos_str)
text(0,0.025,info_str)

% get the best partition pt
min_neg = prctile(neg_dists,0.1);
max_pos = prctile(pos_dists,99.9);
if min_neg>=max_pos
    best_pt = mean([min_neg,max_pos]);
    FP = 0;
    FN = 0;
else
    pts = 99:0.05:100;
    pts = prctile(pos_dists,pts);
    FPs = sum(neg_dists<pts)/numel(neg_dists);
    FNs = sum(pos_dists>pts)/numel(pos_dists);
    [min_total_miss,id] = min(FPs+50*FNs);
    best_pt = pts(id);
    FP = FPs(id);
    FN = FNs(id);
end
best_pt_str = "\downarrow best_pt:"+num2str(best_pt)+newline+"FP: "+num2str(FP)+newline+"50x FN: "+num2str(FN);
text(best_pt,0.04,best_pt_str)

saveas(fig,sprintf('%s.jpg',opts.net.experiment_root));
