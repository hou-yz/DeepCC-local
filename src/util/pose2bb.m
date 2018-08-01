function [ bb ] = pose2bb( pose, renderThreshold )
%POSETOBB Summary of this function goes here
%   Detailed explanation goes here

% Template pose
ref_pose = [0   0; ... %nose
       0,   23; ... % neck
       28   23; ... % rshoulder
       39   66; ... %relbow
       45  108; ... %rwrist
       -28   23; ... % lshoulder
       -39   66; ... %lelbow
       -45  108; ... %lwrist
       20  106; ... %rhip
       20  169; ... %rknee
       20  231; ... %rankle
       -20  106; ... %lhip
       -20  169; ... %lknee
       -20  231; ... %lankle
       5   -7; ... %reye
       11   -8; ... %rear
       -5   -7; ... %leye
       -11   -8; ... %lear
       ];
   
% Template bounding box   
ref_bb = [-50, -15; ...%left top
            50, 240];  % right bottom
        
pose = reshape(pose,[3,18])';
valid = pose(:,1) ~= 0  & pose(:,2) ~=0 & pose(:,3) >= renderThreshold;

if sum(valid) < 2
    bb = [0 0 0 0];
    return;
end

points_det = pose(valid,[1 2]);
points_reference = ref_pose(valid,:);

% 1a) Compute minimum enclosing rectangle

base_left = min(points_det(:,1));
base_top = min(points_det(:,2));
base_right = max(points_det(:,1));
base_bottom = max(points_det(:,2));

% 1b) Fit pose to template

% Find transformation parameters
M = size(points_det,1);
B = points_det(:);
A = [points_reference(:,1) zeros(M,1) ones(M,1)  zeros(M,1); ...
     zeros(M,1)  points_reference(:,2) zeros(M,1)  ones(M,1)];

params = A \ B;
M = 2;
A2 = [ref_bb(:,1) zeros(M,1) ones(M,1)  zeros(M,1); ...
zeros(M,1) ref_bb(:,2) zeros(M,1)  ones(M,1)];

% Visualize
% C = A*params;
% figure(2);
% clf('reset');
% scatter(B(1:end/2),-B(end/2+1:end),'b'); axis equal; hold on
% scatter(C(1:end/2),-C(end/2+1:end),'r+'); axis equal; hold on
result = A2 * params;

fit_left = min(result([1,2]));
fit_top = min(result([3,4]));
fit_right = max(result([1,2]));
fit_bottom = max(result([3,4]));

% 2. Fuse bounding boxes
left = min(base_left,fit_left);
top = min(base_top,fit_top);
right = max(base_right,fit_right);
bottom = max(base_bottom,fit_bottom);

left = left*1920;
top = top*1080;
right = right*1920;
bottom = bottom*1080;

height = bottom - top + 1;
width = right - left + 1;

bb = [left, top, width, height];
    




