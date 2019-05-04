clc
clear
opts = get_opts_aic();

for scene = opts.seqs
    for iCam = opts.cams_in_scene(scene)
        gt          = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));
        labeled_det = load(sprintf('%s/%s/S%02d/c%03d/gt/gt.txt', opts.dataset_path, opts.folder_by_scene{scene}, scene, iCam));