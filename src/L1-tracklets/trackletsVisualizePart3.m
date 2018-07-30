%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW CLUSTERED DETECTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ids = unique(identities);
    
    for kkk = 1:length(ids);
        
        id = ids(kkk);
        
        mask = identities == id;
        mask2 = ismember(currentDetectionsIDX,spatialGroupObservations.*mask);
        rows = find(currentDetectionsIDX .* mask2);
        
        figure(2);
        scatter(detectionCenters(rows,1), detectionCenters(rows,2), 'fill');
        hold on;
        
    end
    
    pause(0.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%