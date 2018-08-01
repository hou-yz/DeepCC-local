function [allMets, metsBenchmark, metsMultiCam] = evaluate(opts)

evalNames   = {'trainval', 'trainval-mini', 'test-easy', 'test-hard'};
seqMap      = sprintf('DukeMTMCT-%s.txt', evalNames{opts.sequence});
eval_folder = [opts.experiment_root, filesep, opts.experiment_name, filesep, opts.eval_dir];
gt_folder   = [opts.dataset_path, 'ground_truth'];
[allMets, metsBenchmark, metsMultiCam] = evaluateTracking(seqMap, eval_folder, gt_folder, 'DukeMTMCT',opts.dataset_path);
