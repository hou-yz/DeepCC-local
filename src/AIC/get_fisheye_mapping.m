function camera_pos_mapping = get_fisheye_mapping(opts)
%GET_FISHEYE_MAPPING Summary of this function goes here
%   Detailed explanation goes here
camera_pos_mapping = cell(1,40);

format long
for iCam = [5,35]
    param = opts.projection{iCam};
    if iCam == 5
        x_range = [42.5272,42.5279];
        y_range = [-90.7259,-90.7250];
    elseif iCam == 35
        x_range = [42.4995,42.5000];
        y_range = [-90.6965,-90.6960];
    end
    interpole = 10;
    x = linspace(x_range(1),x_range(2),interpole);     y = linspace(y_range(1),y_range(2),interpole);
    [x,y] = meshgrid(x,y);
    x = reshape(x,1,interpole^2);    y = reshape(y,1,interpole^2);

    gps_points = [x;y;ones(1,interpole^2)];
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
    
    
    [image_points,index] = unique(round(image_points'),'rows');
    gps_points   = gps_points(1:2,index)';
    camera_pos_mapping{iCam} = struct('pixel',image_points,'gps',gps_points);
end
end

