function [ newbb, newpose ] = scale_bb( bb, pose, scalingFactor )
% Scales bounding box by scaling factor

newbb([1,2]) = bb([1,2]) - 0.5*(scalingFactor-1) * bb([3,4]);
newbb([3,4]) = bb([3,4]) * scalingFactor;

% X, Y, strength
newpose = reshape(pose,3,18)';

% Scale to original bounding box
newpose(:,1) = (newpose(:,1) - bb(1)/1920.0) / (bb(3)/1920.0);
newpose(:,2) = (newpose(:,2) - bb(2)/1080.0) / (bb(4)/1080.0);

% Scale to stretched bounding box
newpose(:,1) = (newpose(:,1) + 0.5*(scalingFactor-1))/scalingFactor;
newpose(:,2) = (newpose(:,2) + 0.5*(scalingFactor-1))/scalingFactor;

% Return in the original format
newpose(newpose(:,3)==0,[1 2]) = 0;
newpose = newpose';
newpose = newpose(:);
newpose = newpose';
