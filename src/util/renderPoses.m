function img = renderPoses( img, poses )
% Renders OpenPose detections in Matlab
POSE_COCO_PAIRS = 1 + [1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,   2,16,  5,17] ;
POSE_COCO_COLORS_RENDER_GPU = ...
    [255,     0,    85; ...
    255,     0,     0; ...
    255,    85,     0; ...
    255,   170,     0; ...
    255,   255,     0; ...
    170,   255,     0; ...
    85,   255,     0; ...
    0,   255,     0; ...
    0,   255,    85; ...
    0,   255,   170; ...
    0,   255,   255; ...
    0,   170,   255; ...
    0,    85,   255; ...
    0,     0,   255; ...
    255,     0,   170; ...
    170,     0,   255; ...
    255,     0,   255; ...
    85,     0,   255];
renderThreshold = 0.05;

%     // Model-Dependent Parameters
%     // COCO
%     const std::map<unsigned int, std::string> POSE_COCO_BODY_PARTS {
%         {0,  "Nose"},
%         {1,  "Neck"},
%         {2,  "RShoulder"},
%         {3,  "RElbow"},
%         {4,  "RWrist"},
%         {5,  "LShoulder"},
%         {6,  "LElbow"},
%         {7,  "LWrist"},
%         {8,  "RHip"},
%         {9,  "RKnee"},
%         {10, "RAnkle"},
%         {11, "LHip"},
%         {12, "LKnee"},
%         {13, "LAnkle"},
%         {14, "REye"},
%         {15, "LEye"},
%         {16, "REar"},
%         {17, "LEar"},
%         {18, "Background"}
% };

%   Detailed explanation goes here
for iPose = 1:size(poses,1)
    pose = poses(iPose,:);
    %         % draw circles
    %         for iPart = 1:18
    %             hold on;
    %             plot(1920*pose(3*iPart-2), 1080*pose(3*iPart-1));
    %         end
    
    % draw lines
    for iPair = 1:length(POSE_COCO_PAIRS)/2
        
        if pose(3*POSE_COCO_PAIRS(2*iPair-1) - 0) < renderThreshold || ...
                pose(3*POSE_COCO_PAIRS(2*iPair) - 0) < renderThreshold || ...
                pose(3*POSE_COCO_PAIRS(2*iPair-1) - 2) == 0 || ...
                pose(3*POSE_COCO_PAIRS(2*iPair) - 2) == 0 || ...
                pose(3*POSE_COCO_PAIRS(2*iPair-1) - 1) == 0 || ...
                pose(3*POSE_COCO_PAIRS(2*iPair) - 1) == 0
            continue;
        end
        
        hold on;
        pt1 = [size(img,2)*pose(3*POSE_COCO_PAIRS(2*iPair-1) - 2), size(img,1)*pose(3*POSE_COCO_PAIRS(2*iPair-1) - 1)];
        pt2 = [size(img,2)*pose(3*POSE_COCO_PAIRS(2*iPair) - 2),   size(img,1)*pose(3*POSE_COCO_PAIRS(2*iPair) - 1)];
        color = POSE_COCO_COLORS_RENDER_GPU(mod(iPair-1,18)+1,:);
%         img = cv.line(img, pt1, pt2, 'Thickness', 2, 'Color', color);
        minx = max(1, floor(min(pt1(1),pt2(1))));
        miny = max(1, floor(min(pt1(2),pt2(2))));
        maxx = min(1920, ceil(max(pt1(1),pt2(1))));
        maxy = min(1080, ceil(max(pt1(2),pt2(2))));
        subimg = img(miny:maxy,minx:maxx,:);
        subimg = cv.line(subimg, pt1 - [minx, miny], pt2-[minx, miny], 'Thickness', 2, 'Color', color);
        img(miny:maxy,minx:maxx,:) = subimg;
    end
    
end

end

