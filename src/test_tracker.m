function [DukeSCT, DukeMCT] = test_tracker(opts,compute_L1,compute_L2,compute_L3)
    if compute_L1
        compute_L1_tracklets(opts);
    end
    if compute_L2
        compute_L2_trajectories(opts);
        opts.eval_dir = 'L2-trajectories';
        [allMets, metsBenchmark, metsMultiCam] = evaluate(opts);
    end
    if compute_L3
        compute_L3_identities(opts);
        opts.eval_dir = 'L3-identities';
        [allMets, metsBenchmark, metsMultiCam] = evaluate(opts);
    end
    DukeSCT = metsBenchmark(1:3);
    DukeMCT = DukeMCT;
end
