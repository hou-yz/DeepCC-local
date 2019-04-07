function image_points = gps2image(opts,gps_points,iCam)
%IMAGE2GPS Summary of this function goes here
%   Detailed explanation goes here
param = opts.projection{iCam};
format long
if isempty(param.intrinsic)
    gps_points = [gps_points,ones(size(gps_points,1),1)]';
    camera_points = param.homography*gps_points;
    image_points = camera_points(1:2,:)./camera_points(3,:);
    image_points = image_points';
else
    % image_points = intrinsic * camera_points = intrinsic * homography * world_points
    gps_points = [gps_points,ones(size(gps_points,1),1)]';
    camera_points = param.intrinsic * param.homography * gps_points;
    camera_points = camera_points(1:2,:)./camera_points(3,:);
    
    x_prime  = camera_points(1,:)';
    y_prime  = camera_points(2,:)';
    k1 = param.distortion(1); k2 = param.distortion(2);
    p1 = param.distortion(3); p2 = param.distortion(4);
    coeff_x = [k1*(x_prime.^2 + y_prime.^2), zeros(size(x_prime)), x_prime.^2, -x_prime.^3];
    coeff_y = [k1*(x_prime.^2 + y_prime.^2), zeros(size(y_prime)), y_prime.^2, -y_prime.^3];
    x = zeros(size(x_prime)); y = zeros(size(y_prime));
    
    for i = 1:length(x_prime)
        r = roots(coeff_x(i,:)); x(i) = r(imag(r)==0);
        r = roots(coeff_y(i,:)); y(i) = r(imag(r)==0);
    end
    
    image_points = [x,y];
end
end
