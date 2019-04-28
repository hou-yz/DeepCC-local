clear
clc

opts = get_opts_aic();
opts.tracklets.window_width = 10000;
opts.experiment_name = 'aic_label_det';
create_experiment_dir(opts);
opts.feature_dir = 'det_features_ide_basis_train_10fps_lr_5e-2_ssd512_trainval';

scene = 4;
fig = 1;

for iCam = opts.cams_in_scene{scene}
    % read
    gt = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));
%     det = load(sprintf('%s/%s/scene%d_cam%d.txt', opts.experiment_root, opts.experiment_name,scene,iCam));
    % tune
    gt(:,2) = gt(:,2)+1000;
    
    for startFrame = 1:opts.tracklets.window_width:7000
        endFrame = startFrame+opts.tracklets.window_width-1;
        window = startFrame:endFrame;
        % data
        in_window_gt = gt(ismember(gt(:,1),window),:);
%         det_lines = ismember(det(:,1),window);
%         in_window_det = det(det_lines,:);
        in_window_gt = [in_window_gt(:,1:2),in_window_gt(:,3:4)+0.5*in_window_gt(:,5:6)];
%         in_window_det = [in_window_det(:,1:2),in_window_det(:,3:4)+0.5*in_window_det(:,5:6)];
        
        %% imshow
%         h = figure(mod(fig,2)+1);        
        figure(1);
        fig = fig+1;
        clf('reset');
        i = opts.reader.getFrame(iCam,startFrame);
        imshow(i);
        pause(1)
%         axis on;
        hold on;
%         %% det
%         if ~isempty(in_window_det)
%             scatter(in_window_det(:,3),in_window_det(:,4),[],in_window_det(:,2),'filled','o');
%             det_ids = unique(in_window_det(:,2));
%             for i = 1:length(det_ids)
%                 det_id = det_ids(i);
%                 centers = in_window_det(in_window_det(:,2)==det_id,3:4);
%                 first_center = centers(1,:);
%                 text(first_center(1),first_center(2),sprintf('%d',det_id),'Color','red','FontSize',14);
%             end
%                 
%         end
        %% gt 
        if ~isempty(in_window_gt)
            scatter(in_window_gt(:,3),in_window_gt(:,4),[],in_window_gt(:,2),'filled','d');
            hold on
%             sz = size(i);
%             truesize(h,sz(1:2))
%             saveas(fig,fullfile(sprintf('%s/aic_label_det/background', opts.experiment_root),sprintf('c%02d.jpg',iCam)))
        end
    end
    
%     dlmwrite(sprintf('%s/%s/scene%d_cam%d.txt', ...
%         opts.experiment_root, ...
%         opts.experiment_name, ...
%         scene,iCam), ...
%         labeled_det, 'delimiter', ' ', 'precision', 6);
%         
end