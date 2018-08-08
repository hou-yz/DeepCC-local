function [ C ] = world2map( W )

image_points = [307.4323  469.2366; 485.2483  708.9507];
world_points = [0 0; 24.955 32.85];

diff = image_points(2,:) - image_points(1,:);

scale = diff ./ world_points(2,:);
trans = image_points(1,:);

C(:,1) = W(:,1)*scale(1) + trans(1);
C(:,2) = W(:,2)*scale(2) + trans(2);

end

