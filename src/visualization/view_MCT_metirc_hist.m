clc
clear

type = 'mid';
mct=1;

opts = get_opts();

opts.sequence = 8;
opts.experiment_name = '1fps_train_IDE_40';
opts.tracklets.window_width = 40;
opts.trajectories.window_width = 150;
opts.identities.window_width = 3000;

opts.appear_model_name = '1fps_train_IDE_40/GT/model_param_L3_2400.mat';
appear_model_param_L = load(fullfile('src','hyper_score/logs',opts.appear_model_name));
opts.appear_model_name = '1fps_train_IDE_40/GT/model_param_L3_inf.mat';
appear_model_param_G = load(fullfile('src','hyper_score/logs',opts.appear_model_name));

    sequence_window   = opts.sequence_intervals{opts.sequence};
    startFrame       = sequence_window(1);
    endFrame         = sequence_window(end);
    
filename = sprintf('%s/ground_truth/%s/all_trajectories_%s.mat',opts.dataset_path,opts.experiment_name,opts.sequence_names{opts.sequence});
%% 
if ~exist(filename)
all_tracklets = [];
all_trajectories = [];
for iCam = 1:4
    
    load(sprintf('%s/ground_truth/%s/tracklets%d_trainval.mat',opts.dataset_path,opts.experiment_name,iCam))
    iCams=num2cell(iCam*ones(1,length(tracklets)));
    [tracklets.iCam]=iCams{:};
    start_fs=num2cell(local2global(opts.start_frames(iCam), [tracklets.startFrame]));
    end_fs=num2cell(local2global(opts.start_frames(iCam), [tracklets.endFrame]));
    [tracklets.startFrame]=start_fs{:};
    [tracklets.endFrame]=end_fs{:};
    tracklets(([tracklets.endFrame]>endFrame) + ([tracklets.startFrame]<startFrame)>0)=[];
    pids = [tracklets.id]';
    tracklets(pids==-1)=[];
    pids = unique([tracklets.id]);
    all_tracklets = [all_tracklets,tracklets];
    
    for i = 1:length(pids)
        pid = pids(i);
        target_tracklets = tracklets([tracklets.id]==pid);
        clear target_trajectories
        target_trajectories(1) = target_tracklets(1);
        for j = 2:length(target_tracklets)
            if target_tracklets(j).startFrame > target_trajectories(end).endFrame + opts.trajectories.window_width% || length(target_trajectories(end).data)>60
                target_trajectories(end+1) = target_tracklets(j);
            else
                target_trajectories(end).features = [target_trajectories(end).features; target_tracklets(j).features];
                target_trajectories(end).data = [target_trajectories(end).data; target_tracklets(j).data];
                target_trajectories(end).feature = mean(cell2mat(target_trajectories(end).features));
                target_trajectories(end).endFrame = target_tracklets(j).endFrame;
            end
        end
        if length(unique([target_trajectories.id]))>1
            pid
        end
        if length(target_trajectories)>1
            pid
        end
        all_trajectories = [all_trajectories,target_trajectories];
    end
end

save(filename, 'all_trajectories','-v7.3');
end
load(filename)

all_metric_score_L = [];
all_metric_score_G = [];
    
    startFrame = sequence_window(1);
    endFrame   = sequence_window(1) + opts.identities.window_width - 1;
    while startFrame <= sequence_window(end)
        % Display loop state
        clc; fprintf('Window %d...%d\n', startFrame, endFrame);

        % traj
        in_window_tracklet = all_trajectories(([all_trajectories.endFrame]<endFrame) .* ([all_trajectories.startFrame]>startFrame)>0);
        % Update loop range
        startFrame = endFrame   - opts.identities.window_width/2;
        endFrame   = startFrame + opts.identities.window_width;
        if isempty(in_window_tracklet)
            continue
        end
        in_window_pids = [in_window_tracklet.id]';
        unique_pids = unique(in_window_pids);
        num_traj_per_pid = zeros(size(unique_pids));
        for i = 1:length(unique_pids)
            pid = unique_pids(i);
            num_traj_per_pid(i) = numel([in_window_tracklet([in_window_tracklet.id]==pid).iCam]);
        end
        
        if ~sum(num_traj_per_pid>1)
            continue
        end
        
        
%         figure(3)
%         for i = 1:length(unique_pids)
%             pid = unique_pids(i);
%             pid_tracklets = in_window_tracklet([in_window_tracklet.id]==pid);
%             tracklet = pid_tracklets(1);
%             iCam = tracklet.iCam;
%             frame = tracklet.data(round(end/2),1);
%             bbox = tracklet.data(round(end/2),3:6);
% %             fig=show_bbox(opts,iCam,frame,bbox);
% %             subplot(1,length(unique_pids),i)
% %             imshow(fig)
% %             if ismember(i,[16:25])
% %                 continue
% %             end
% %             in_window_tracklet([in_window_tracklet.id]==pid)=[];
%         end
        
        
        in_window_feat = reshape([in_window_tracklet.feature]',length(in_window_tracklet(1).feature),[])';
        % score
        distanceL = getHyperScore(in_window_feat,appear_model_param_L,opts.soft,0,0,0);
        distanceG = getHyperScore(in_window_feat,appear_model_param_G,opts.soft,0,0,0);
        in_window_sameLabels  = pdist2(in_window_pids, in_window_pids) == 0;
        all_metric_score_L = [all_metric_score_L;[distanceL(:),in_window_sameLabels(:)]];
        all_metric_score_G = [all_metric_score_G;[distanceG(:),in_window_sameLabels(:)]];
        
        
        if startFrame == 196540
            view_tsne_2
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


