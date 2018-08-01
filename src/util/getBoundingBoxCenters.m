function [ centers ] = getBoundingBoxCenters( boundingBoxes )
% Returns the centers of the bounding boxes provided in the format:
% left, top, right, botom

centers = [ boundingBoxes(:,1) + 0.5*boundingBoxes(:,3), boundingBoxes(:,2) + 0.5* boundingBoxes(:,4)];




