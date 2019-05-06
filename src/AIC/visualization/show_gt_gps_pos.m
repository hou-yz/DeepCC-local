clear
clc

opts = get_opts_aic();
scene = 4;
all_gt = [];

for iCam = opts.cams_in_scene{scene}
    data = load(sprintf('%s/train/S%02d/c%03d/gt/gt_gps.txt', opts.dataset_path, scene, iCam));
    all_gt = [all_gt;data];
end

all_gt = sortrows(all_gt, [2,9,8]);

pids = unique(all_gt(:,2));

for i = 1:length(pids)
    figure(1)
    clf('reset');
    axis equal
    hold on;
    pid = pids(i);
    lines = all_gt(all_gt(:,2) == pid,:);
    for iCam = opts.cams_in_scene{scene}
        icam_pos = lines(lines(:,8)==iCam,10:11);
        if isempty(icam_pos)
            continue
        end
        scatter(icam_pos(:,1),icam_pos(:,2),'filled')
    end
    pause(1)
end
