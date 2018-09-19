clear
clc
for iCam = 1:8
    opts = get_opts();
    data = readtable('src/triplet-reid/data/duke_test.csv', 'Delimiter',',');

    opts.net.experiment_root = 'experiments/fc256_30fps_separate_icam_fake';
    labels = data.Var1;
    paths  = data.Var2;
    ids =  contains(paths,sprintf('_c%d_',iCam));
    labels = labels(ids,:);
    paths = paths(ids,:);
    %% Compute features
    % separate feat
    features = h5read(fullfile(opts.net.experiment_root, sprintf('features_icam%d.h5',iCam)),'/emb');
    % global feat
%     features = h5read(fullfile(opts.net.experiment_root, 'features.h5'),'/emb');
    features = features';
    features = features(ids,:);
    dist = pdist2(features,features);
    %% Visualize distance distribution
    same_label = triu(pdist2(labels,labels) == 0,1);
    different_label = triu(pdist2(labels,labels) ~= 0);
    pos_dists = dist(same_label);
    neg_dists = dist(different_label);
    pos_99_5th = prctile(pos_dists,99.5);
    neg_5th = prctile(neg_dists,5);
    mid = mean([mean(pos_dists),mean(neg_dists)]);
    
    fig = figure;
    hold on;
    histogram(pos_dists,100,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,100,'Normalization','probability','FaceColor','r');
    title('Normalized distribution of distances among positive and negative pairs');
    legend('Positive','Negative');
    neg_str = ['\downarrow dist_P less than the 5th percentile dist_N: ',num2str(sum(pos_dists<neg_5th)/length(pos_dists))];
    mid_str = ['\downarrow dist_P less than the mid value: ',num2str(sum(pos_dists<mid)/length(pos_dists))];
    pos_str = "\downarrow dist_P 99.5th";
    info_str = "dist_P 99.5th: "+num2str(pos_99_5th)+newline+"dist_N 5th: "+num2str(neg_5th)+newline+"mid value: "+num2str(mid);
    
    text(neg_5th,0.025,neg_str)
    text(mid,0.02,mid_str)
    text(pos_99_5th,0.03,pos_str)
    text(0,0.025,info_str)
    
    % get the best partition pt
    min_neg = prctile(neg_dists,0.1);
    max_pos = prctile(pos_dists,99.9);
    if min_neg>=max_pos
        best_pt = mean([min_neg,max_pos]);
        FP = 0;
        FN = 0;
    else
        pts = 99.5:0.05:100;
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

    saveas(fig,sprintf('%s%d.jpg',opts.net.experiment_root,iCam));
end