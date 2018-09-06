load('D:\MATLAB\Data\DukeMTMC\ground_truth\trainval.mat')
% [camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY]
ids = unique(trainData(:,2));
consecute_cam = zeros(length(ids),8);
cam_matrix = zeros(8,8);
for i = 1:length(ids)
    id = ids(i);
    lines = trainData(trainData(:,2)==id,:);
    cams = unique(lines(:,1));
    consecute_cam(i,cams) = 1;
    cam_matrix(cams,cams) = cam_matrix(cams,cams) + 1;
end