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
else
    % image_points = intrinsic * camera_points = intrinsic * homography * world_points
    % without distortion
    camera_points = param.homography * gps_points;
    camera_points = camera_points(1:2,:)./camera_points(3,:);
    x  = camera_points(1,:)'; y  = camera_points(2,:)';
    % add distortion 
    k = param.distortion;
    r = sqrt(x.^2 + y.^2);
%     theta = 2/pi* atan(sqrt(x.^2 + y.^2));
    theta = atan(r);
    rho = theta + k(1)*theta.^3;
    x_prime = rho./r.*x; y_prime = rho./r.*y;
    camera_points = [x_prime';y_prime';ones(1,size(gps_points,2))];
    
    image_points = param.intrinsic * camera_points;
    image_points = image_points(1:2,:)./image_points(3,:);
end
image_points = image_points';
end
