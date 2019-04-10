function gps_points = image2gps(opts,image_points,iCam)
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
    pixel = opts.fisheye_mapping{iCam}.pixel;
    gps = opts.fisheye_mapping{iCam}.gps;
    dist = pdist2(image_points(:,1),pixel(:,1)) + pdist2(image_points(:,2),pixel(:,2));
    [~,index] = min(dist,[],2);
    gps_points = gps(index,:);
%     % with distortion
%     camera_points = param.intrinsic \ [image_points,ones(size(image_points,1),1)]';
%     camera_points = (camera_points(1:2,:)./camera_points(3,:))';
%     x_prime  = camera_points(:,1);    y_prime  = camera_points(:,2);
%     % remove distortion
%     % assume theta is normalized to [0,1]
%     rho = sqrt(x_prime.^2 + y_prime.^2);
%     k = param.distortion;
%     coeff = [ones(length(camera_points),1)*[k(1),0,1],-rho];
%     theta = zeros(length(camera_points),1);
%     for i = 1:10%length(camera_points)
%         s = roots(coeff(i,:));  
%         theta(i) = s(s<1 & s>0);
%     end
%     r = tan(theta * pi/2);
%     x = x_prime.^2 .* r.^2 ./ rho.^2;    y = y_prime.^2 .* r.^2 ./ rho.^2;
%     % without distortion
%     camera_points = [x,y,ones(length(image_points),1)]';
%     gps_points = param.homography \ camera_points;
%     gps_points = (gps_points(1:2,:)./gps_points(3,:))';
end
gps_points = (gps_points - opts.world_center{opts.trainval_scene_by_icam(iCam)}) * opts.world_scale;
end
