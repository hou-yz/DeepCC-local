function gps_points = image2gps(opts,image_points,iCam)
%IMAGE2GPS Summary of this function goes here
%   Detailed explanation goes here
param = opts.projection{iCam};
format long
if isempty(param.intrinsic)
    camera_points = [image_points,ones(size(image_points,1),1)]';
    gps_points = param.homography\camera_points;
    gps_points = gps_points(1:2,:)./gps_points(3,:);
else
    x  = image_points(:,1)';
    y  = image_points(:,2)';
    r2 = x.^2 + y.^2;
    xy = x .* y;
    k1 = param.distortion(1); k2 = param.distortion(2);
    p1 = param.distortion(3); p2 = param.distortion(4);
    x_prime = x.*(1 + k1*r2 + k2*r2.^2) + 2*p1*xy + p2*(r2 + 2*x.^2);
    y_prime = y.*(1 + k1*r2 + k2*r2.^2) + p1*(r2 + 2*y.^2) + 2*p2*xy;
    
    % image_points = intrinsic * camera_points = intrinsic * homography * world_points
    camera_points = param.intrinsic\[x_prime;y_prime;ones(1,size(image_points,1))];
    gps_points = (param.homography\camera_points);
    gps_points = gps_points(1:2,:)./gps_points(3,:);

end
gps_points = gps_points';
end
