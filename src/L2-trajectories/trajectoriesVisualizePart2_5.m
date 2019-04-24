%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW CLUSTERED DETECTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2);
hold on;
    ids = unique(identities);
    
    for kkk = 1:length(ids)
        id = ids(kkk);
        groupped_tracklets = consider_tracklets(identities == id);
        gourpped_data = vertcat(groupped_tracklets.data);
        detectionCenters = getBoundingBoxCenters(gourpped_data(:,3:6));
        
        scatter(detectionCenters(:,1), detectionCenters(:,2), 'fill');
        hold on;
    end
    
    pause(0.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%