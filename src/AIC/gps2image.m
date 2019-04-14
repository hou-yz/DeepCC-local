function image_points = gps2image(opts,gps_points,iCam)
%IMAGE2GPS Summary of this function goes here
%   Detailed explanation goes here
param = opts.projection{iCam};
format long
gps_points = gps_points / opts.world_scale  + opts.world_center{opts.trainval_scene_by_icam(iCam)}; 
gps_points = [gps_points,ones(size(gps_points,1),1)]';

if isempty(param.intrinsic)
    camera_points = param.homography * gps_points;
    image_points = camera_points(1:2,:)./camera_points(3,:);
    image_points = image_points';
else
    % image_points = intrinsic * camera_points = intrinsic * homography * world_points
    % without distortion
    camera_points = param.homography * gps_points;
    camera_points = round(camera_points(1:2,:)./camera_points(3,:));
    camera_points = camera_points';
    
    pix_x = opts.fisheye_mapping{iCam}.undistort2pix_x;
    pix_y = opts.fisheye_mapping{iCam}.undistort2pix_y;
    linear_idx = sub2ind(size(pix_x),camera_points(:,1),camera_points(:,2));
    image_points = [pix_x(linear_idx), pix_y(linear_idx)];
    
end
end
