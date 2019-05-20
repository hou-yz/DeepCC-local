function [thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type,dataset)
% data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',','); % reid

% opts.net.experiment_root = 'experiments/fc256_6fps_epoch45';
%% Compute features

% cam, pid, frame, 256-dim feat
features = [];
if dataset == 0
    for iCam = 1:8
    tmp_features = h5read(fullfile(opts.net.experiment_root, sprintf('features%d.h5',iCam)),'/emb');
    tmp_features = tmp_features';
    sequence_window   = opts.sequence_intervals{opts.sequence};
    sequence_window = global2local(opts.start_frames(iCam), sequence_window);
    in_time_range_ids = ismember(tmp_features(:,3),sequence_window);
    features = [features;tmp_features(in_time_range_ids,:)];
    end
elseif dataset == 1
    features = h5read(sprintf('%s/%s/all_seq_feat.h5',opts.feature_dir,opts.net.experiment_root),'/emb')';
else
    for scene = opts.seqs{opts.sequence}
    for iCam = opts.cams_in_scene{scene}
        tmp_features = h5read(fullfile(opts.net.experiment_root, sprintf('features%d.h5',iCam)),'/emb');
        tmp_features = tmp_features';
        features = [features;tmp_features];
    end
    end
end

features(features(:,1)==-1,:)=[];
% pooling
if dataset == 2
    pooling = 4;
else
    pooling = 10;
end

if dataset == 1
    labels = features(1:pooling:length(features),2)+features(1:pooling:length(features),1)*1000;
    cams = [];
else
    labels = features(1:pooling:length(features),2);
    cams = features(1:pooling:length(features),1);
end
features = single(features(1:pooling:length(features),4:end));
dist = pdist2(features,features);
clear features
%% figure settings
    fig = figure;
    % Enlarge figure to full screen.
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.7, 0.7]);
    hold on;

    %% Visualization for CROSS CAMERA
    same_label = triu(pdist2(labels,labels) == 0,1);
    different_label = triu(pdist2(labels,labels) ~= 0);
    pos_dists = double(dist(same_label));
    neg_dists = double(dist(different_label));
    %%
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title('Normalized distribution of pair-distances');
    legend('Cross-camera:  positive', 'Cross-camera:  negative')
    set(gca,'YTickLabel',[]);
    
    %%
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    
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
    
    % decide thres
    if strcmp(type,'mid')
    thres_uni = mid;
    else
    thres_uni = best_pt;
    end
    diff_p_uni = thres_uni - m(1);
    diff_n_uni = m(2) - thres_uni;
    
disp("thres:  "+num2str(thres_uni,'%.2f '))
disp("diff_p: "+num2str(diff_p_uni,'%.2f '))
disp("diff_n: "+num2str(diff_n_uni,'%.2f '))
    best_pt_str = "cross-camera" +newline+ "\downarrow thres: "+num2str(thres_uni,'%2.2f');
    text(best_pt,0.03,best_pt_str)
%     info_str = "5% dist_N: "+num2str(neg_5th,'%.2f')+newline+"mid:       "+num2str(mid,'%.2f')+newline+"best\_pt:  "+num2str(best_pt,'%.2f');
%     text(mid,0.05,info_str)
%     best_pt_str = "\downarrow dist_P less than the best\_pt: "+num2str(sum(pos_dists<best_pt)/length(pos_dists)*100,'%.2f')+"%";
%     text(best_pt,0.03,best_pt_str)
%     dist_str = "E[d_N-d_P]: "+num2str(diff,'%.2f')+newline+"0.5dist: "+num2str(diff/2,'%.2f')+newline+"diff_P: "+num2str(diff_p_uni,'%.2f')+newline+"diff_N: "+num2str(diff_n_uni,'%.2f');
%     text(0,0.05,dist_str)

    %% Visualization for WITHIN CAMERA
    same_label = triu(pdist2(labels,labels) == 0 & pdist2(cams,cams) == 0,1);
    different_label = triu(pdist2(labels,labels) ~= 0 & pdist2(cams,cams) == 0);
    pos_dists = double(dist(same_label));
    neg_dists = double(dist(different_label));
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'g');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','y');
    title('Normalized distribution of pair-distances');
    legend('Cross-camera:  positive', 'Cross-camera:  negative', 'Within-camera: positive', 'Within-camera: negative')
    set(gca,'YTickLabel',[]);
    
    %%
    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    
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
    
    % decide thres
    if strcmp(type,'mid')
    thres_uni = mid;
    else
    thres_uni = best_pt;
    end
    diff_p_uni = thres_uni - m(1);
    diff_n_uni = m(2) - thres_uni;
    
disp("thres:  "+num2str(thres_uni,'%.2f '))
disp("diff_p: "+num2str(diff_p_uni,'%.2f '))
disp("diff_n: "+num2str(diff_n_uni,'%.2f '))
    best_pt_str = "within-camera" +newline+ "\downarrow thres: "+num2str(thres_uni,'%2.2f');
    text(best_pt,0.01,best_pt_str)

    hold off
    
    saveas(fig,sprintf('%s.jpg',opts.net.experiment_root));
    

end