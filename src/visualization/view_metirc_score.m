clc
clear

type = 'mid';


opts = get_opts();
filename = 'L3_12000';
% filename = 'L2_75';
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','1fps_train_IDE_40',sprintf('results_%s_train_Inf.h5',filename)),'/emb');
appear_score = all_data(1,:)';
motion_score = all_data(2,:)';
target = all_data(3,:)';

    same_label = target==1;
    different_label = target==0;
    pos_dists = double(appear_score(same_label));
    neg_dists = double(appear_score(different_label));
    % pooling
%     pooling = floor(length(neg_dists)/length(pos_dists));
%     neg_dists = neg_dists(1:pooling:length(neg_dists));
    %%
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
%     pos_halves = pos_dists-m(1);
%     neg_halves = neg_dists-m(2);
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    %% dist
    fig = figure;
    % Enlarge figure to full screen.
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.7, 0.7]);
%     fig.PaperPositionMode = 'auto';
    subplot(1,2,1)
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title('Normalized distribution of pair-distances');
    errorbar(m(1),0.3,s(1),'horizontal','b-o')
    errorbar(m(2),0.3,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    stat_str_P = "mean_P:"+num2str(m(1),'%.3f')+newline+"std_P:"+num2str(s(1),'%.3f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.3f')+newline+"std_N:"+num2str(s(2),'%.3f');
%     neg_str = "\downarrow dist_P less than the 5% dist_N: "+num2str(sum(pos_dists<neg_5th)/length(pos_dists)*100,'%.3f')+"%";
    mid_str = "\downarrow dist_P less than mid: "+num2str(sum(pos_dists<mid)/length(pos_dists)*100,'%.3f')+"%";
%     text(neg_5th,0.025,neg_str)
%     text(mid,0.2,mid_str)
    text(m(1),0.25,stat_str_P)
    text(m(2),0.25,stat_str_N)
    
    max_neg = prctile(neg_dists,99.9);
    min_pos = prctile(pos_dists,0.01);
    if max_neg<=min_pos
        best_pt = mean([max_neg,min_pos]);
        FP = 0;
        FN = 0;
    else
        pts = linspace(min_pos,max_neg,100);
        FPs = sum(neg_dists>pts)/numel(neg_dists);
        FNs = sum(pos_dists<pts)/numel(pos_dists);
        [min_total_miss,id] = min(FPs+1*FNs);
        best_pt = pts(id);
        FP = FPs(id);
        FN = FNs(id);
    end
    info_str ="mid:       "+num2str(mid,'%.3f')+newline+"best\_pt:  "+num2str(best_pt,'%.3f');
    text(mid,0.2,info_str)
    
    % decide thres
    if strcmp(type,'mid')
    thres_uni = mid;
    else
    thres_uni = best_pt;
    end
    diff_p_uni = m(1)-thres_uni ;
    diff_n_uni = thres_uni-m(2);
    dist_str = "E[d_N-d_P]: "+num2str(diff,'%.3f')+newline+"0.5dist: "+num2str(diff/2,'%.3f')+newline+"diff_P: "+num2str(diff_p_uni,'%.3f')+newline+"diff_N: "+num2str(diff_n_uni,'%.3f');
    text(0,0.5,dist_str)

    hold off
    
    %% score
    subplot(1,2,2)
    pos_dists = -(thres_uni-pos_dists)/diff_p_uni;
    neg_dists = -(thres_uni-neg_dists)/diff_n_uni;
    
    o0pos = sum(pos_dists>0)/length(pos_dists)*100;
    o0neg = sum(neg_dists<0)/length(neg_dists)*100;
    o5pos = sum(pos_dists>0.5)/length(pos_dists)*100;
    o5neg = sum(neg_dists<-0.5)/length(neg_dists)*100;
    
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title(sprintf('Apparance score: %s',type));
    errorbar(m(1),0.3,s(1),'horizontal','b-o')
    errorbar(m(2),0.3,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.3f')+newline+"std_P:"+num2str(s(1),'%.3f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.3f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.3f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.3f')+newline+"std_N:"+num2str(s(2),'%.3f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.3f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.3f')+"%";
    text(m(1),0.25,stat_str_P)
    text(m(2),0.25,stat_str_N)
    
    saveas(fig,sprintf('metric_score_%s.jpg',filename));
    
    
disp("thres:  "+num2str(thres_uni,'%.3f '))
disp("diff_p: "+num2str(diff_p_uni,'%.3f '))
disp("diff_n: "+num2str(diff_n_uni,'%.3f '))
    