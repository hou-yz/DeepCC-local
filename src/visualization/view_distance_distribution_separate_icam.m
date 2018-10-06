function [threshold_s,diff_p_s,diff_n_s]=view_distance_distribution_separate_icam(opts, type,unified_model,thres_uni,diff_p_uni,diff_n_uni)

threshold_s = zeros(1,8);
diff_p_s = zeros(1,8);
diff_n_s = zeros(1,8);
for iCam = 1:8
    data = readtable('src/visualization/file_list.csv', 'Delimiter',','); % gt@1fps
%     data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',','); % reid

%     opts.net.experiment_root =  'experiments/fc256_30fps_separate_icam'; %'experiments/fc256_6fps_epoch45';%
    labels = data.Var1;
    paths  = data.Var2;
    ids =  contains(paths,sprintf('_c%d_',iCam));
    labels = labels(ids,:);
    paths = paths(ids,:);
    %% Compute features
    if unified_model
    % global feat
    features = h5read(fullfile(opts.net.experiment_root, 'features.h5'),'/emb');
    else
    % separate feat
    features = h5read(fullfile(opts.net.experiment_root, sprintf('features_icam%d.h5',iCam)),'/emb');
    end
    features = features';
    features = features(ids,:);
    dist = pdist2(features,features);
    %% Visualize distance distribution
    same_label = triu(pdist2(labels,labels) == 0,1);
    different_label = triu(pdist2(labels,labels) ~= 0);
    pos_dists = dist(same_label);
    neg_dists = dist(different_label);
    % pooling
%     pooling = floor(length(neg_dists)/length(pos_dists));
%     neg_dists = neg_dists(1:pooling:length(neg_dists));
    
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
    thres = mid;
    else
    thres = best_pt;
    end
    diff_p = thres - m(1);
    diff_n = m(2) - thres;
    dist_str = "E[d_N-d_P]: "+num2str(diff,'%.2f')+newline+"0.5diff: "+num2str(diff/2,'%.2f')+newline+"diff_P: "+num2str(diff_p,'%.2f')+newline+"diff_N: "+num2str(diff_n,'%.2f');
    text(0,0.05,dist_str)

    hold off
    
    %% appear score
    subplot(1,2,2)
    pos_dists_1 = (thres-pos_dists)/diff_p;
    neg_dists_1 = (thres-neg_dists)/diff_n;
    
    threshold_s(iCam) = thres;
    diff_p_s(iCam) = diff_p;
    diff_n_s(iCam) = diff_n;
    
    o0pos = sum(pos_dists_1>0)/length(pos_dists_1)*100;
    o0neg = sum(neg_dists_1<0)/length(neg_dists_1)*100;
    o5pos = sum(pos_dists_1>0.5)/length(pos_dists_1)*100;
    o5neg = sum(neg_dists_1<-0.5)/length(neg_dists_1)*100;
    
    m = [mean(pos_dists_1),mean(neg_dists_1)];
    s = [std(pos_dists_1),std(neg_dists_1)];
    
    hold on;
    histogram(pos_dists_1,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists_1,100,'Normalization','probability','FaceColor','r');
    title(['Apparance score: ',type]);
    errorbar(m(1),0.07,s(1),'horizontal','b-o')
    errorbar(m(2),0.07,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','northwest');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.2f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.2f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.2f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.2f')+"%";
    text(m(1),0.065,stat_str_P)
    text(m(2),0.065,stat_str_N)
    
    if thres_uni~=0
    %% unified appear score
    
    
    pos_dists_2 = (thres_uni-pos_dists)/diff_p_uni;
    neg_dists_2 = (thres_uni-neg_dists)/diff_n_uni;
    
    o0pos = sum(pos_dists_2>0)/length(pos_dists_2)*100;
    o0neg = sum(neg_dists_2<0)/length(neg_dists_2)*100;
    o5pos = sum(pos_dists_2>0.5)/length(pos_dists_2)*100;
    o5neg = sum(neg_dists_2<-0.5)/length(neg_dists_2)*100;
    
    m = [mean(pos_dists_2),mean(neg_dists_2)];
    s = [std(pos_dists_2),std(neg_dists_2)];
    
    hold on;
    histogram(pos_dists_2,100,'Normalization','probability', 'FaceColor', 'g');
    histogram(neg_dists_2,100,'Normalization','probability','FaceColor','y');
    errorbar(m(1),-0.02,s(1),'horizontal','g-o')
    errorbar(m(2),-0.02,s(2),'horizontal','y-o')
    legend('Pos\_sep','Neg\_sep','stats_P\_sep','stats_N\_sep','Pos\_uni','Neg\_uni','stats_P\_uni','stats_N\_uni','Location','northwest');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.2f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.2f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.2f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.2f')+"%";
    text(m(1),-0.01,stat_str_P)
    text(m(2),-0.01,stat_str_N)
        
    
    elseif ~strcmp(type,'mid')
    %% mid_value score
    thres_mid = mid;
    diff_p_mid = diff/2;
    diff_n_mid = diff/2;
        
    pos_dists_2 = (thres_mid-pos_dists)/diff_p_mid;
    neg_dists_2 = (thres_mid-neg_dists)/diff_n_mid;
    
    o0pos = sum(pos_dists_2>0)/length(pos_dists_2)*100;
    o0neg = sum(neg_dists_2<0)/length(neg_dists_2)*100;
    o5pos = sum(pos_dists_2>0.5)/length(pos_dists_2)*100;
    o5neg = sum(neg_dists_2<-0.5)/length(neg_dists_2)*100;
    
    m = [mean(pos_dists_2),mean(neg_dists_2)];
    s = [std(pos_dists_2),std(neg_dists_2)];
    
    hold on;
    histogram(pos_dists_2,100,'Normalization','probability', 'FaceColor', 'g');
    histogram(neg_dists_2,100,'Normalization','probability','FaceColor','y');
    errorbar(m(1),-0.02,s(1),'horizontal','g-o')
    errorbar(m(2),-0.02,s(2),'horizontal','y-o')
    legend('Pos\_sep','Neg\_sep','stats_P\_sep','stats_N\_sep','Pos\_mid','Neg\_mid','stats_P\_mid','stats_N\_mid','Location','northwest');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.2f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.2f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.2f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.2f')+"%";
    text(m(1),-0.01,stat_str_P)
    text(m(2),-0.01,stat_str_N)
        
        
    end
    
    
    saveas(fig,sprintf('%s_%d.jpg',opts.net.experiment_root,iCam));
end
disp("thres:  "+num2str(threshold_s,'%.2f '))
disp("diff_p: "+num2str(diff_p_s,'%.2f '))
disp("diff_n: "+num2str(diff_n_s,'%.2f '))
