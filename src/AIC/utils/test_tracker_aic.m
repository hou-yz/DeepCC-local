function [DukeSCT, DukeMCT] = test_tracker_aic(opts,compute_L1,compute_L2,compute_L3)
    if compute_L1
        compute_L1_tracklets_aic(opts);
    end
    if compute_L2
        compute_L2_trajectories_aic(opts);
        opts.eval_dir = 'L2-trajectories';
        [allMets, metsBenchmark, metsMultiCam] = evaluate(opts);
    end
    if compute_L3
        compute_L3_identities_aic(opts);
        opts.eval_dir = 'L3-identities';
        [allMets, metsBenchmark, metsMultiCam] = evaluate(opts);
    end
    DukeSCT = metsBenchmark(1:3);
    DukeMCT = metsMultiCam;
end
