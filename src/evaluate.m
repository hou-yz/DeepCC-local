function [allMets, metsBenchmark, metsMultiCam] = evaluate(opts)
if opts.dataset == 0 % duke
    evalNames   = {'trainval', 'trainval-mini', 'test-easy', 'test-hard','','','','val'};
    seqMap      = sprintf('DukeMTMCT-%s.txt', evalNames{opts.sequence});
    eval_folder = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
    gt_folder   = [opts.dataset_path, filesep, 'ground_truth'];
    [allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqMap, eval_folder, gt_folder, 'DukeMTMCT',opts.dataset_path);
elseif opts.dataset == 1 % mot
    evalNames   = {'trainval', 'trainval-mini', 'test-easy', 'test-hard','','','train','val'};
    seqMap      = sprintf('MOT16-%s.txt', evalNames{opts.sequence});
    eval_folder = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
    gt_folder   = [opts.dataset_path, filesep, 'train', filesep];
    [allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqMap, eval_folder, gt_folder, 'MOT16',opts.dataset_path);
elseif opts.dataset == 2 % aic
    evalNames   = {'trainval', '', 'test-easy', '','','','train','val'};
    seqMap      = sprintf('AIC19-%s.txt', evalNames{opts.sequence});
    eval_folder = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
    gt_folder   = [opts.dataset_path, filesep, 'ground_truth'];
    [allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqMap, eval_folder, gt_folder, 'AIC19',opts.dataset_path);
end
end