clc
clear

type = 'mid';
mct=0;

opts = get_opts();

% opts.visualize = 1;

opts.sequence = 8;
opts.experiment_name = '1fps_train_IDE_40';
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 600;
opts.identities.window_width = 6000;

opts.appear_model_name = '1fps_train_IDE_40/GT/model_param_L2_600.mat';
appear_model_param_L = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
opts.appear_model_name = '1fps_train_IDE_40/GT/model_param_L3_inf.mat';
appear_model_param_G = load(fullfile('src','hyper_score/logs',opts.appear_model_name));

all_metric_score_L = [];
all_metric_score_G = [];
%% Computes single-camera trajectories from tracklets
for iCam = 1:8
    sequence_window   = opts.sequence_intervals{opts.sequence};
    startFrame       = global2local(opts.start_frames(iCam), sequence_window(1));
    endFrame         = global2local(opts.start_frames(iCam), sequence_window(end));
    
    filename =sprintf('%s/ground_truth/%s/tracklets%d_trainval.mat',opts.dataset_path,opts.experiment_name,iCam);
    load(filename)
    tmp=num2cell(iCam*ones(1,length(tracklets)));
    [tracklets.iCam]=tmp{:};
    start_fs = [tracklets.startFrame];
    end_fs = [tracklets.endFrame];
    tracklets((end_fs>endFrame) + (start_fs<startFrame)>0)=[];
    pids = [tracklets.id]';
    tracklets(pids==-1)=[];
    start_fs = [tracklets.startFrame];
    end_fs = [tracklets.endFrame];
    
    startFrame = global2local(opts.start_frames(iCam), sequence_window(1));
    endFrame   = global2local(opts.start_frames(iCam), sequence_window(1) + opts.trajectories.window_width - 1);
    while startFrame <= global2local(opts.start_frames(iCam), sequence_window(end))
        % Display loop state
        clc; fprintf('Cam: %d - Window %d...%d\n', iCam, startFrame, endFrame);

        in_window_tracklet = tracklets((end_fs<endFrame) .* (start_fs>startFrame)>0);
        % Update loop range
        startFrame = endFrame   - opts.trajectories.window_width/2;
        endFrame   = startFrame + opts.trajectories.window_width;
        if isempty(in_window_tracklet)
            continue
        end
        in_window_pids = [in_window_tracklet.id]';
        in_window_feat = reshape([in_window_tracklet.feature]',length(in_window_tracklet(1).feature),[])';
        % score
        distanceL = getHyperScore(in_window_feat,appear_model_param_L,opts.soft,0,0,0);
        distanceG = getHyperScore(in_window_feat,appear_model_param_G,opts.soft,0,0,0);
        in_window_sameLabels  = pdist2(in_window_pids, in_window_pids) == 0;
        all_metric_score_L = [all_metric_score_L;[distanceL(:),in_window_sameLabels(:)]];
        all_metric_score_G = [all_metric_score_G;[distanceG(:),in_window_sameLabels(:)]];
        
        
        

        num_traj_per_pid = ones(size(in_window_tracklet));
        
        if startFrame == 183498
            view_tsne_2
        end
    end

end

    fig = figure;
    % Enlarge figure to full screen.
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.1, 0.4, 0.9]);
    set(gca, 'FontName', 'Arial')
    %% fig 1 
    subplot(2,1,1);
    hold on;
    pos_dists = all_metric_score_G(all_metric_score_G(:,2)==1,1);
    neg_dists = all_metric_score_G(all_metric_score_G(:,2)==0,1);

    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    histogram(pos_dists,20,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,20,'Normalization','probability','FaceColor','r');
    title(['Normalized distribution of SCT metric scores:',newline,'traditional metric learning']);
    errorbar(m(1),0.5,s(1),'horizontal','b-o')
    errorbar(m(2),0.5,s(2),'horizontal','r-o')
    legend('Positive','Negative','Location','west');
    stat_str_P = "mean_P:"+num2str(m(1),'%.3f')+newline+"std_P:"+num2str(s(1),'%.3f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.3f')+newline+"std_N:"+num2str(s(2),'%.3f');
    text(m(1),0.5,stat_str_P)
    text(m(2),0.5,stat_str_N)
    
    
        FP = sum(neg_dists>0)/length(all_metric_score_G)*100;
        FN = sum(pos_dists<0)/length(all_metric_score_G)*100;
        FPFN_str = "FP:"+num2str(FP,'%.3f')+newline+"FN:"+num2str(FN,'%.3f');
        text(0,0.5,FPFN_str)
    
    axis([-6 4 0 0.8])
    hold off
    
    
    %% fig 2 
    subplot(2,1,2);
    hold on;
    pos_dists = all_metric_score_L(all_metric_score_L(:,2)==1,1);
    neg_dists = all_metric_score_L(all_metric_score_L(:,2)==0,1);

    m = [mean(pos_dists),mean(neg_dists)];
    s = [std(pos_dists),std(neg_dists)];
    mid = mean(m);
    diff = mean(neg_dists)-mean(pos_dists);
    histogram(pos_dists,20,'Normalization','probability', 'FaceColor', 'b');
    histogram(neg_dists,20,'Normalization','probability','FaceColor','r');
    title(['Normalized distribution of SCT metric scores:',newline,'TLML']);
    errorbar(m(1),0.7,s(1),'horizontal','b-o')
    errorbar(m(2),0.7,s(2),'horizontal','r-o')
    legend('Positive','Negative','Location','west');
    stat_str_P = "mean_P:"+num2str(m(1),'%.3f')+newline+"std_P:"+num2str(s(1),'%.3f');
    stat_str_N = "mean_N:"+num2str(m(2),'%.3f')+newline+"std_N:"+num2str(s(2),'%.3f');
    text(m(1),0.7,stat_str_P)
    text(m(2),0.7,stat_str_N)
    
    
        FP = sum(neg_dists>0)/length(all_metric_score_L)*100;
        FN = sum(pos_dists<0)/length(all_metric_score_L)*100;
        FPFN_str = "FP:"+num2str(FP,'%.3f')+newline+"FN:"+num2str(FN,'%.3f');
        text(0,0.5,FPFN_str)
    
    axis([-6 4 0 0.8])
    hold off

