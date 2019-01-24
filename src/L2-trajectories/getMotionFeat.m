function feat = getMotionFeat(tracklets, iCam)

    [~, ~, startpoint, endpoint, intervals, ~, ~] = getTrackletFeatures(tracklets);
    [startpoint, ~, ~] = image2world( startpoint, iCam );
    [endpoint, ~, ~]   = image2world( endpoint, iCam );
    velocity        = (endpoint-startpoint)./(intervals(:,2)-intervals(:,1));
    intervals       = local2global(opts.start_frames(iCam),intervals);
    centerFrame     = round(mean(intervals,2));
    centers         = 0.5 * (endpoint + startpoint);

    feat = [centerFrame,centers,centers,velocity,velocity];
    
end