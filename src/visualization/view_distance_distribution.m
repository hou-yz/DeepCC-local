function [thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type)
data = readtable('src/visualization/file_list.csv', 'Delimiter',','); % gt@1fps
% data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',','); % reid

% opts.net.experiment_root = 'experiments/fc256_6fps_epoch45';
labels = data.Var1;
paths  = data.Var2;
%% Compute features
features = h5read(fullfile(opts.net.experiment_root, 'features.h5'),'/emb');
features = features';
% pooling
pooling = 4;
labels = labels(1:pooling:length(labels),:);
features = features(1:pooling:length(features),:);
dist = pdist2(features,features);
%% Visualize distance distribution
    same_label = triu(pdist2(labels,labels) == 0,1);
    different_label = triu(pdist2(labels,labels) ~= 0);
    pos_dists = dist(same_label);
    neg_dists = dist(different_label);
    %%
    neg_dists = randsample(neg_dists,length(pos_dists));
    %%
    neg_5th = prctile(neg_dists,5);
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
%     pos_halves = pos_dists-m(1);
%     neg_halves = neg_dists-m(2);
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    %%
    fig = figure;
    % Enlarge figure to full screen.
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.7, 0.7]);
%     fig.PaperPositionMode = 'auto';
    subplot(1,2,1)
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title('Normalized distribution of pair-distances');
    errorbar(m(1),0.07,s(1),'horizontal','b-o')
    errorbar(m(2),0.07,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','southeast');
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f');
    neg_str = "\downarrow dist_P less than the 5% dist_N: "+num2str(sum(pos_dists<neg_5th)/length(pos_dists)*100,'%.2f')+"%";
    mid_str = "\downarrow dist_P less than mid: "+num2str(sum(pos_dists<mid)/length(pos_dists)*100,'%.2f')+"%";
    text(neg_5th,0.025,neg_str)
    text(mid,0.02,mid_str)
    text(m(1),0.065,stat_str_P)
    text(m(2),0.065,stat_str_N)
    
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
        [min_total_miss,id] = min(FPs+1*FNs);
        best_pt = pts(id);
        FP = FPs(id);
        FN = FNs(id);
    end
    info_str = "5% dist_N: "+num2str(neg_5th,'%.2f')+newline+"mid:       "+num2str(mid,'%.2f')+newline+"best\_pt:  "+num2str(best_pt,'%.2f');
    text(mid,0.05,info_str)
    best_pt_str = "\downarrow dist_P less than the best\_pt: "+num2str(sum(pos_dists<best_pt)/length(pos_dists)*100,'%.2f')+"%";
    text(best_pt,0.03,best_pt_str)
    
    % decide thres
    if strcmp(type,'mid')
    thres_uni = mid;
    else
    thres_uni = best_pt;
    end
    diff_p_uni = thres_uni - m(1);
    diff_n_uni = m(2) - thres_uni;
    dist_str = "E[d_N-d_P]: "+num2str(diff,'%.2f')+newline+"0.5dist: "+num2str(diff/2,'%.2f')+newline+"diff_P: "+num2str(diff_p_uni,'%.2f')+newline+"diff_N: "+num2str(diff_n_uni,'%.2f');
    text(0,0.05,dist_str)

    hold off
    
    %%
    subplot(1,2,2)
    pos_dists = (thres_uni-pos_dists)/diff_p_uni;
    neg_dists = (thres_uni-neg_dists)/diff_n_uni;
    
    o0pos = sum(pos_dists>0)/length(pos_dists)*100;
    o0neg = sum(neg_dists<0)/length(neg_dists)*100;
    o5pos = sum(pos_dists>0.5)/length(pos_dists)*100;
    o5neg = sum(neg_dists<-0.5)/length(neg_dists)*100;
    
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title(['Apparance score',type]);
    errorbar(m(1),0.07,s(1),'horizontal','b-o')
    errorbar(m(2),0.07,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','southeast');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.2f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.2f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.2f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.2f')+"%";
    text(m(1),0.065,stat_str_P)
    text(m(2),0.065,stat_str_N)
    
    
    saveas(fig,sprintf('%s.jpg',opts.net.experiment_root));
    
disp("thres:  "+num2str(thres_uni,'%.2f '))
disp("diff_p: "+num2str(diff_p_uni,'%.2f '))
disp("diff_n: "+num2str(diff_n_uni,'%.2f '))

end