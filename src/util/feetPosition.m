function [ feet ] = feetPosition( boxes )
%FEETPOSITION Summary of this function goes here
%   Detailed explanation goes here

x = boxes(:,1) + boxes(:,3)/2;
y = boxes(:,2) + boxes(:,4);
feet = [x, y];

end

