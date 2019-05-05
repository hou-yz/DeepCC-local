function valid = getAICValidDetections(iCam, detections, imageROI)
    if ismember(iCam, [1:9,16,22,25,26,27,34:36,50])
        ignore_size = [100,80];
    elseif ismember(iCam, 10:15)
        ignore_size = [120,100];
    else
        ignore_size = [100,80];
    end
    ignore_size = ignore_size - 40;
    frames = unique(detections(:,1));
    % Flags valid OpenPose detections
    valid = true(size(detections,1),1);
    
    for i = 1:length(frames)
    frame           = frames(i);
    line_ids        = find(detections(:,1) == frame);
    bboxs           = detections(line_ids, [3, 4, 5, 6]);
    feet_pos        = feetPosition(bboxs);
    %% NMS
    ious            = bboxOverlapRatio(bboxs,bboxs);
    ious_min        = bboxOverlapRatio(bboxs,bboxs,'Min');
    merging         = ious>0.9 & ious_min>0.9; 
    [rows,columns]  = find(triu(merging,1));
    valid(columns)  = false;
    %% ROI
    inds            = sub2ind( size(imageROI), uint64(feet_pos(:,2)), uint64(feet_pos(:,1)));
    valid(line_ids(imageROI(inds) == 0)) = false;
    valid(line_ids(bboxs(:,3)<ignore_size(1) | bboxs(:,4)<ignore_size(2))) = false;
    end
end