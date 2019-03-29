function render_results_aic()
% Creates a movie for each camera view to help visualize errors
% Requires that single-camera results exists in experiment folder L2-trajectories

% IDTP - Green
% IDFP - Blue
% IDFN - Black
tail_colors = [0 0 1; 0 1 0; 0 0 0];
tail_size = 100;

colors = distinguishable_colors(1000);

opts = get_opts_aic();

folder = 'video-results';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);

% Load ground truth
load(fullfile(opts.dataset_path, 'ground_truth', 'train.mat'));

% Render one video per camera
for i = 1:length(opts.seqs)
    
    iCam = opts.seqs(i);
    
    readerVideo = MyVideoReader_aic(iCam);
    
    % Create video
    filename = sprintf('%s/%s/%s/cam%d_%s',opts.experiment_root, opts.experiment_name, folder, iCam, opts.sequence_names{opts.sequence});
    video = VideoWriter(filename);
    video.FrameRate = 10;
    open(video);
    
    % Load result
    resdata = dlmread(sprintf('%s/%s/L2-trajectories/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.sequence_names{opts.sequence}));
    
    % Load relevant ground truth
    gtdata = trainData;
    filter = gtdata(:,1) == iCam;
    gtdata = gtdata(filter,:);
    gtdata = gtdata(:,2:end);
    gtdata(:,[1 2]) = gtdata(:,[2 1]);
    gtdata = sortrows(gtdata,[1 2]);
    gtMat = gtdata;
    
    resMat = resdata;
    
    % Compute error types
    [gtMatViz, predMatViz] = error_types(gtMat,resMat,0.5,0);
    gtMatViz = sortrows(gtMatViz, [1 2]);
    predMatViz = sortrows(predMatViz, [1 2]);
    
    detections      = load(sprintf('%s/train/S%02d/c%03d/det/det_yolo3.txt', opts.dataset_path, opts.scene_by_icam(iCam), iCam));
    start_frame     = detections(1, 1);
    end_frame       = detections(end, 1);
    
    %     tic
    for iFrame = start_frame:end_frame
        %IDFP
        %         if mod(iFrame,100)==1
        %             t_100 = toc
        %             tic
        %             fprintf('Cam %d:  %d/%d\n', iCam, iFrame, global2local(opts.start_frames(iCam),sequence_interval(end)));
        %         end
        fprintf('Cam %d:  %d/%d\n', iCam, iFrame, end_frame);
        image  = readerVideo.getFrame(iCam, iFrame);
        
        rows        = find(predMatViz(:, 1) == iFrame);
        identities  = predMatViz(rows, 2);
        positions   = [predMatViz(rows, 3),  predMatViz(rows, 4), predMatViz(rows, 5), predMatViz(rows, 6)];
        
        if ~isempty(positions)
            image = insertObjectAnnotation(image,'rectangle', ...
                positions, identities,'TextBoxOpacity', 0.8, 'FontSize', 16, 'Color', 255*colors(identities,:) );
        end
        
        % Tail Pred
        rows = find((predMatViz(:, 1) <= iFrame) & (predMatViz(:,1) >= iFrame - tail_size));
        identities = predMatViz(rows, 2);
        
        feetposition = feetPosition(predMatViz(rows,3:6));
        is_TP = predMatViz(rows,end);
        current_tail_colors = [];
        for kkk = 1:length(is_TP)
            current_tail_colors(kkk,:) = tail_colors(is_TP(kkk)+1,:);
        end
        
        circles = feetposition;
        circles(:,3) = 3;
        image = insertShape(image,'FilledCircle',circles,'Color', current_tail_colors*255);
        
        % IDFN
        rows = find((gtMatViz(:, 1) <= iFrame) & (gtMatViz(:,1) >= iFrame - tail_size));
        feetposition = feetPosition(gtMatViz(rows,3:6));
        
        is_TP = gtMatViz(rows,end);
        current_tail_colors = [];
        for kkk = 1:length(is_TP)
            current_tail_colors(kkk,:) = tail_colors(3-is_TP(kkk),:);
        end
        circles = feetposition;
        circles(:,3) = 3;
        image = insertShape(image,'FilledCircle',circles(~is_TP,:),'Color', current_tail_colors(~is_TP,:)*255);
        image = insertText(image,[0 0], sprintf('Cam %d - Frame %d',iCam, iFrame),'FontSize',20);
        image = insertText(image,[0 40; 60 40; 120 40], {'IDTP', 'IDFP','IDFN'},'FontSize',20,'BoxColor',{'green','blue','black'},'TextColor',{'white','white','white'});
        
        writeVideo(video, image);
        
    end
    close(video);
    
end

