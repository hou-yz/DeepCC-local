clear
clc

opts = get_opts_aic();
opts.tracklets.window_width = 10;
opts.experiment_name = 'aic_label_det';
create_experiment_dir(opts);
opts.feature_dir = 'det_features_ide_basis_train_10fps_lr_5e-2_ssd512_trainval';

scene = 1;

for iCam = opts.cams_in_scene{scene}
    % read    
    det = load(sprintf('%s/%s/S%02d/c%03d/det/det_%s.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam, opts.detections));
    features = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb');
    % tune
    labeled_det = det;
    labeled_det(:,11:12) = labeled_det(:,3:4) + 0.5*labeled_det(:,5:6);
    labeled_det(:,2) = 10000;
    features = features';
    
    for startFrame = 1:opts.tracklets.window_width:7000
        endFrame = startFrame+opts.tracklets.window_width-1;
        window = startFrame:endFrame;
        % data
        det_lines = ismember(det(:,1),window);
        in_window_det = det(det_lines,:);
        in_window_feat = features(ismember(features(:,2),window),3:end);
        in_window_det = [in_window_det(:,1:2),in_window_det(:,3:4)+0.5*in_window_det(:,5:6)];
        
        %% det
        
        % Increase number of groups until no group is too large to solve         
        num_appearance_groups = floor(length(in_window_det)/opts.tracklets.window_width);
        if num_appearance_groups == 0
            continue
        end
        while true
        appearanceGroups    = kmeans(in_window_feat, num_appearance_groups, 'emptyaction', 'singleton', 'Replicates', 10);
        uid = unique(appearanceGroups);
        freq = [histc(appearanceGroups(:),uid)];
        largestGroupSize = max(freq);
        if largestGroupSize <= 1.5*opts.tracklets.window_width
            break
        end
        num_appearance_groups = num_appearance_groups+1;
        end
        labeled_det(det_lines,2) = appearanceGroups+max(labeled_det(:,2));
        
    end
    
    dlmwrite(sprintf('%s/%s/scene%d_cam%d.txt', ...
        opts.experiment_root, ...
        opts.experiment_name, ...
        scene,iCam), ...
        labeled_det, 'delimiter', ' ', 'precision', 6);
        
end