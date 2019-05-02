clear
clc

opts = get_opts_aic();
opts.tracklets.window_width = 10;
opts.experiment_name = 'aic_zju';
opts.sequence = 8;



for scene = opts.seqs{opts.sequence}
for iCam = opts.cams_in_scene{scene}
    opts.current_camera = iCam;
    labeled_data = [];
    % Initialize
    load(fullfile(opts.experiment_root, opts.experiment_name, 'L1-tracklets', sprintf('tracklets%d_%s.mat',iCam,opts.sequence_names{opts.sequence})));
    for i = 1:length(tracklets)
        tracklets(i).realdata(:,2) = i;
        tracklets(i).realdata(:,7:8) = [];
        
        labeled_data = [labeled_data;tracklets(i).realdata];
        
        frame = tracklets(i).realdata(1,1);
        image = opts.reader.getFrame(scene,iCam,frame);
        bbox = tracklets(i).realdata(1,3:6);
        det_image=get_bb(image, bbox);
        imwrite(det_image,fullfile(sprintf('%s/aic_label_det/label_bboxs', opts.experiment_root),sprintf('c%d_f%04d_%04d.jpg',iCam,frame,i+10000)))
    end
    labeled_data(:,2) = labeled_data(:,2)+10000;
    
    dlmwrite(sprintf('%s/aic_label_det/scene%d_cam%d.txt', ...
        opts.experiment_root, ...
        scene,iCam), ...
        labeled_data, 'delimiter', ' ', 'precision', 6);
        
end
end

    
    