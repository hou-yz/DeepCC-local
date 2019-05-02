function gps_points = image2gps(opts, image_points, scene, iCam)
%IMAGE2GPS Summary of this function goes here
%   Detailed explanation goes here
param = opts.projection{iCam};
format long
if isempty(param.intrinsic)
    camera_points = [image_points,ones(size(image_points,1),1)]';
    gps_points = param.homography \ camera_points;
    gps_points = (gps_points(1:2,:)./gps_points(3,:))';
else
%     image_points = intrinsic * camera_points = intrinsic * homography * world_points

    pix_x = opts.fisheye_mapping{iCam}.undistort2pix_x;
    pix_y = opts.fisheye_mapping{iCam}.undistort2pix_y;   
    
    undistorted_x = 1:size(pix_x,1);undistorted_y = 1:size(pix_x,2);
    [undistorted_x,undistorted_y] = meshgrid(undistorted_x,undistorted_y);
    undistorted_x = undistorted_x'; undistorted_y = undistorted_y';
    
    pix_x = reshape(pix_x,[],1);
    pix_y = reshape(pix_y,[],1);
    undistorted_x = reshape(undistorted_x,[],1);
    undistorted_y = reshape(undistorted_y,[],1);

    camera_points = zeros(size(image_points));
    
    for i = 1:100:length(image_points)
    line_ids = i:min(i+99,length(image_points));
    dist = pdist2(image_points(line_ids,1),pix_x) + pdist2(image_points(line_ids,2),pix_y);
    [~,index] = min(dist,[],2);
    camera_points(line_ids,:) = [undistorted_x(index),undistorted_y(index)];
    end
    
    camera_points = [camera_points,ones(size(image_points,1),1)]';
    
    gps_points = param.homography \ camera_points;
    gps_points = (gps_points(1:2,:)./gps_points(3,:))';
end
gps_points = (gps_points - opts.world_center{scene}) * opts.world_scale;
end
