clc
clear
opts = get_opts();
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','1fps_train_PCB_40','target_data.h5'),'/emb');
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
    errorbar(m(1),0.7,s(1),'horizontal','b-o')
    errorbar(m(2),0.7,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f');
%     neg_str = "\downarrow dist_P less than the 5% dist_N: "+num2str(sum(pos_dists<neg_5th)/length(pos_dists)*100,'%.2f')+"%";
    mid_str = "\downarrow dist_P less than mid: "+num2str(sum(pos_dists<mid)/length(pos_dists)*100,'%.2f')+"%";
%     text(neg_5th,0.025,neg_str)
    text(mid,0.2,mid_str)
    text(m(1),0.65,stat_str_P)
    text(m(2),0.65,stat_str_N)
    
    % decide thres
    thres_uni = mid;
    diff_p_uni = m(1)-thres_uni ;
    diff_n_uni = thres_uni-m(2);
    dist_str = "E[d_N-d_P]: "+num2str(diff,'%.2f')+newline+"0.5dist: "+num2str(diff/2,'%.2f')+newline+"diff_P: "+num2str(diff_p_uni,'%.2f')+newline+"diff_N: "+num2str(diff_n_uni,'%.2f');
    text(0,0.5,dist_str)

    hold off
    
    %%
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
    title(['Apparance score: mid']);
    errorbar(m(1),0.7,s(1),'horizontal','b-o')
    errorbar(m(2),0.7,s(2),'horizontal','r-o')
    legend('Positive','Negative','stats_P','stats_N','Location','east');
    
    
    stat_str_P = "mean_P:"+num2str(m(1),'%.2f')+newline+"std_P:"+num2str(s(1),'%.2f')+newline+"score_P > 0.5: "+num2str(o5pos,'%.2f')+"%"+newline+"score_P > 0:    "+num2str(o0pos,'%.2f')+"%";
    stat_str_N = "mean_N:"+num2str(m(2),'%.2f')+newline+"std_N:"+num2str(s(2),'%.2f')+newline+"score_N < -0.5: "+num2str(o5neg,'%.2f')+"%"+newline+"score_N < -0:    "+num2str(o0neg,'%.2f')+"%";
    text(m(1),0.65,stat_str_P)
    text(m(2),0.65,stat_str_N)
    
    
    
disp("thres:  "+num2str(thres_uni,'%.2f '))
disp("diff_p: "+num2str(diff_p_uni,'%.2f '))
disp("diff_n: "+num2str(diff_n_uni,'%.2f '))
    