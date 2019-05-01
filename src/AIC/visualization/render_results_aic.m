% Creates a movie for each camera view to help visualize errors
% Requires that single-camera results exists in experiment folder L2-trajectories

% IDTP - Green
% IDFP - Blue
% IDFN - Black
clc
clear

tail_colors = [0 0 1; 0 1 0; 0 0 0];
tail_size = 100;

colors = distinguishable_colors(5000);

opts = get_opts_aic();
opts.experiment_name = 'aic_zju';
opts.sequence = 1;

folder = 'video-results';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);
% Load ground truth
if opts.sequence ~=6
    load(fullfile(opts.dataset_path, 'ground_truth', 'train.mat'));
end
% Render one video per camera
for scene = opts.seqs{opts.sequence}
for i = 1:length(opts.cams_in_scene{scene})
    iCam = opts.cams_in_scene{scene}(i);
    
    % Create video
    filename = sprintf('%s/%s/%s/cam%d_%s',opts.experiment_root, opts.experiment_name, folder, iCam, opts.sequence_names{opts.sequence});
    video = VideoWriter(filename);
    video.FrameRate = 10;
    open(video);
    
    % Load result
    resdata = dlmread(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.sequence_names{opts.sequence}));
    resMat = resdata;
    
    if opts.sequence ~=6
        % Load relevant ground truth
        gtdata = trainData;
        filter = gtdata(:,1) == iCam;
        gtdata = gtdata(filter,:);
        gtdata = gtdata(:,2:end);
        gtdata(:,[1 2]) = gtdata(:,[2 1]);
        gtdata = sortrows(gtdata,[1 2]);
        gtMat = gtdata;
        % Compute error types
        [gtMatViz, predMatViz] = error_types(gtMat,resMat,0.5,0);
        gtMatViz = sortrows(gtMatViz, [1 2]);
        predMatViz = sortrows(predMatViz, [1 2]);
    else
        gtMatViz   = [];
        predMatViz = resMat;
    end
    
    for frame = min(resMat(:, 1)):max(resMat(:, 1))
        %IDFP
        if mod(frame,100)==0
            tic
            fprintf('Cam %d:  %d/%d\n', iCam, frame, max(resMat(:, 1)));
        end
        img  = opts.reader.getFrame(iCam, frame-1);
        img_size    = size(img);
        
        rows        = find(predMatViz(:, 1) == frame);
        identities  = predMatViz(rows, 2);
        positions   = predMatViz(rows, 3:6);
        positions(:,1:2) = min(max(positions(:,1:2),1),img_size(2:-1:1)-1);
        positions(:,3:4) = max(min(positions(:,1:2)+positions(:,3:4),img_size(2:-1:1))-positions(:,1:2),1);
        
        
        if ~isempty(positions) && ~isempty(identities)
            img = insertObjectAnnotation(img,'rectangle', ...
                positions, identities,'TextBoxOpacity', 0.8, 'FontSize', 16, 'Color', 255*colors(identities,:) );
        end
        
        % Tail Pred
        rows = find((predMatViz(:, 1) <= frame) & (predMatViz(:,1) >= frame - tail_size));
        identities = predMatViz(rows, 2);
        feetposition = feetPosition(predMatViz(rows,3:6));
        
        if opts.sequence ~=6
            is_TP = predMatViz(rows,end);
            current_tail_colors = [];
            for kkk = 1:length(is_TP)
                current_tail_colors(kkk,:) = tail_colors(is_TP(kkk)+1,:);
            end
        else
            current_tail_colors = colors(identities,:);
        end
        
        circles = feetposition;
        circles(:,3) = 3;
        img = insertShape(img,'FilledCircle',circles,'Color', current_tail_colors*255);
        
        % IDFN
        if opts.sequence ~=6
            rows = find((gtMatViz(:, 1) <= frame) & (gtMatViz(:,1) >= frame - tail_size));
            feetposition = feetPosition(gtMatViz(rows,3:6));
        
            is_TP = gtMatViz(rows,end);
            current_tail_colors = [];
            for kkk = 1:length(is_TP)
                current_tail_colors(kkk,:) = tail_colors(3-is_TP(kkk),:);
            end
            circles = feetposition;
            circles(:,3) = 3;
            img = insertShape(img,'FilledCircle',circles(~is_TP,:),'Color', current_tail_colors(~is_TP,:)*255);
            img = insertText(img,[0 0], sprintf('Cam %d - Frame %d',iCam, frame),'FontSize',20);
            img = insertText(img,[0 40; 60 40; 120 40], {'IDTP', 'IDFP','IDFN'},'FontSize',20,'BoxColor',{'green','blue','black'},'TextColor',{'white','white','white'});
        end
            
        writeVideo(video, img);
        
    end
    close(video);
    
end
end

