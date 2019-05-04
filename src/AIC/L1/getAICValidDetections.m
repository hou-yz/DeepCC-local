function valid = getAICValidDetections(scene, detections, imageROI)
    if scene == 1 || scene == 2
        ignore_size = 50;
    else
        ignore_size = 70;
    end
    frames = unique(detections(:,1));
    % Flags valid OpenPose detections
    valid = true(size(detections,1),1);
    
    for i = 1:length(frames)
    frame           = frames(i);
    line_ids        = find(detections(:,1) == frame);
    bboxs           = detections(line_ids, [3, 4, 5, 6]);
    center          = [bboxs(:,2) + 0.5*bboxs(:,4), bboxs(:,1) + 0.5*bboxs(:,3)];
        
%     ious            = bboxOverlapRatio(bboxs,bboxs);
%     ious_min        = bboxOverlapRatio(bboxs,bboxs,'Min');
%     merging         = ious>0.9 & ious_min>0.9; 
%     [rows,columns]  = find(triu(merging,1));
%     valid(columns)  = false;

    inds            = sub2ind(size(imageROI), uint64(center(:,1)), uint64(center(:,2)));
    valid(line_ids(imageROI(inds) == 0)) = false;
    valid(line_ids(bboxs(:,3)<ignore_size | bboxs(:,4)<ignore_size)) = false;
    end
end