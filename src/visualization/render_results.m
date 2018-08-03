function render_results(opts)
% Creates a movie for each camera view to help visualize errors
% Requires that single-camera results exists in experiment folder L2-trajectories

% IDTP - Green
% IDFP - Blue
% IDFN - Black
tail_colors = [0 0 1; 0 1 0; 0 0 0];
tail_size = 100;

colors = distinguishable_colors(1000);

folder = 'video-results';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);

% Load ground truth
load(fullfile(opts.dataset_path, 'ground_truth', 'trainval.mat'));

% Render one video per camera
for iCam = 1:opts.num_cam
    
    % Create video
    filename = sprintf('%s/%s/%s/cam%d_%s.mp4',opts.experiment_root, opts.experiment_name, folder, iCam, opts.sequence_names{opts.sequence});
    video = VideoWriter(filename, 'MPEG-4');
    open(video);
    
    % Load result
    predMat = dlmread(sprintf('%s/%s/L2-trajectories/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.sequence_names{opts.sequence}));
    
    sequence_interval = opts.sequence_intervals{opts.sequence};
    
    % Load relevant ground truth
    gtdata = trainData;
    filter = gtdata(:,1) == iCam & ismember(gtdata(:,3) + opts.start_frames(iCam) - 1, sequence_interval);
    gtdata = gtdata(filter,:);
    gtdata = gtdata(:,2:end);
    gtdata(:,[1 2]) = gtdata(:,[2 1]);
    gtdata = sortrows(gtdata,[1 2]);
    gtMat = gtdata;

    % Compute error types
    [gtMatViz, predMatViz] = error_types(gtMat,predMat,0.5,0);
    gtMatViz = sortrows(gtMatViz, [1 2]);
    predMatViz = sortrows(predMatViz, [1 2]);

    for iFrame = global2local(opts.start_frames(iCam), sequence_interval(1)):1:global2local(opts.start_frames(iCam),sequence_interval(end))
        fprintf('Cam %d:  %d/%d\n', iCam, iFrame, global2local(opts.start_frames(iCam),sequence_interval(end)));
        if mod(iFrame,5) >0
            continue;
        end
        image  = opts.reader.getFrame(iCam,iFrame);

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

