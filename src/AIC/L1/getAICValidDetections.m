function valid = getAICValidDetections(scene, detections_in_interval, imageROI)
    if scene == 1 || scene == 2
        ignore_size = 50;
    else
        ignore_size = 70;
    end
    % Flags valid OpenPose detections
    valid = true(size(detections_in_interval,1),1);

    for k = 1:size(detections_in_interval,1)

        bboxDetection = detections_in_interval(k, [3, 4, 5, 6]);
        centerDetection = [bboxDetection(2) + 0.5*bboxDetection(4), bboxDetection(1) + 0.5*bboxDetection(3)];
        
        if imageROI(uint64(centerDetection(1)), uint64(centerDetection(2))) == 0
            valid(k) = false;
        end

        if bboxDetection(3)<ignore_size || bboxDetection(4)<ignore_size
            valid(k) = false;
        end
    end
end