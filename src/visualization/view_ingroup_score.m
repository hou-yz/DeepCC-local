clc
clear
opts = get_opts();
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','1fps_train_IDE_40','hyperGT_L2_train_75.h5'),'/hyperGT');
% iCam, pid, centerFrame, SpaGrpID, pos*2, v*2, 0, 256-dim feat
considered_line = all_data(2,:)~=-1;
all_pids = all_data(2,considered_line)';
spagrp_ids = all_data(4,considered_line)';
all_feats = all_data(10:end,considered_line)';

unique_spa_id = unique(spagrp_ids);
pos_dists = [];
neg_dists = [];
for i = 1:length(unique_spa_id)
    current_spa_id = unique_spa_id(i);
    pids = all_pids(spagrp_ids==current_spa_id);
    feats = all_feats(spagrp_ids==current_spa_id,:);
    dist = pdist2(feats,feats);
    
    same_label = triu(pdist2(pids,pids) == 0,1);
    different_label = triu(pdist2(pids,pids) ~= 0);
    pos_dists = [pos_dists;double(dist(same_label))];
    neg_dists = [neg_dists;double(dist(different_label))];
end
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
%     pos_halves = pos_dists-m(1);
%     neg_halves = neg_dists-m(2);
    fig = figure;
    % Enlarge figure to full screen.
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.7, 0.7]);
%     fig.PaperPositionMode = 'auto';
    subplot(2,1,1)
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
%     title('Normalized distribution of pair-distances');
    errorbar(m(1),0.1,s(1),'horizontal','b-o')
    errorbar(m(2),0.1,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f');
    text(m(1),0.1,stat_str_P)
    text(m(2),0.1,stat_str_N)
    
    hold off
    
    %%
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','1fps_train_IDE_40','hyperGT_L2_train_inf.h5'),'/hyperGT');
% iCam, pid, centerFrame, SpaGrpID, pos*2, v*2, 0, 256-dim feat
considered_line = all_data(2,:)~=-1;
all_pids = all_data(2,considered_line)';
spagrp_ids = all_data(4,considered_line)';
all_feats = all_data(10:end,considered_line)';

unique_spa_id = unique(spagrp_ids);
pos_dists = [];
neg_dists = [];
for i = 1:length(unique_spa_id)
    pooling = 10;
    current_spa_id = unique_spa_id(i);
    pids = all_pids(spagrp_ids==current_spa_id);
    pids = pids(1:pooling:length(pids));
    feats = all_feats(spagrp_ids==current_spa_id,:);
    feats = feats(1:pooling:length(feats),:);
    dist = pdist2(feats,feats);
    
    same_label = triu(pdist2(pids,pids) == 0,1);
    different_label = triu(pdist2(pids,pids) ~= 0);
    pos_dists = [pos_dists;double(dist(same_label))];
    neg_dists = [neg_dists;double(dist(different_label))];
end
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    subplot(2,1,2)
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
%     title('Normalized distribution of pair-distances');
    errorbar(m(1),0.1,s(1),'horizontal','b-o')
    errorbar(m(2),0.1,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f');
    text(m(1),0.1,stat_str_P)
    text(m(2),0.1,stat_str_N)
    
    hold off
