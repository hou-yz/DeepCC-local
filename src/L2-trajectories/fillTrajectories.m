function [ detectionsUpdated ] = fillTrajectories( detections )
% This function adds points by interpolation to the resulting trajectories,
% so that trajectories are complete and results are less influenced by
% false negative detections

detections = sortrows(detections,[1 3 4 5 6]);


detectionsUpdated = detections;

personIDs = unique(detections(:,2));

count = 0;

for i = 1 : length(personIDs)
   
    personID = personIDs( i );
    
    relevantDetections = detections( detections(:,2) == personID, : );
    
    startFrame = min( relevantDetections(:,1));
    endFrame = max( relevantDetections(:,1) );
    
    missingFrames = setdiff( [startFrame:endFrame]', relevantDetections(:,1) );
    
    if isempty(missingFrames)
        
        continue;
        
    end
    
    frameDiff = diff(missingFrames') > 1;
    
    startInd = [1, frameDiff];
    endInd = [frameDiff, 1];
    
    startInd = find(startInd);
    endInd = find(endInd);
    
  
    
    for k = 1:length(startInd)
       
        interpolatedDetections = zeros( missingFrames(endInd(k)) - missingFrames(startInd(k)) + 1 , size(detections,2) );
        
        interpolatedDetections(:,2) = personID;
        interpolatedDetections(:,1) = [ missingFrames(startInd(k)):missingFrames(endInd(k)) ]';
        
        preDetection = detections( (detections(:,2) == personID) .* detections(:,1) == missingFrames(startInd(k)) - 1, :);
        postDetection = detections( (detections(:,2) == personID) .* detections(:,1) == missingFrames(endInd(k)) + 1, :);
        
        
        for c = 3:size(detections,2)
           
            interpolatedDetections(:,c) = linspace(preDetection(c),postDetection(c),size(interpolatedDetections,1));
            
        end
        
        detectionsUpdated = [ detectionsUpdated; interpolatedDetections ];
        
    end
    
    count = count + length( missingFrames );
    
    
    
    

end






