function [ centersWorld, centersView, startpoint, endpoint, intervals, duration, velocity ] = getTrackletFeatures( tracklets )

numTracklets = length(tracklets);

% bounding box centers for each tracklet

centersWorld = cell( numTracklets, 1 );
centersView = cell( numTracklets, 1 );

for i = 1 : numTracklets
    
    detections = cell2mat({tracklets(i).data});
        
    % 2d points
    bb = detections( :, [3,4,5,6,1] );
    x = 0.5*(bb(:,1) + bb(:,3));
    y = 0.5*(bb(:,2) + bb(:,4));
    t = bb(:,5);
    centersView{i} = [x,y,t];

    % 3d points
%     worldcoords = detections(:, [7,8,2] );
%     x = worldcoords(:,1);
%     y = worldcoords(:,2);
%     t = worldcoords(:,3);
%     centersWorld{i} = [x,y,t];
    centersWorld{i} = centersView{i};
    
end

% calculate velocity, direction, for each tracklet

velocity = zeros(numTracklets,2);
duration = zeros(numTracklets,1);
intervals = zeros(numTracklets,2);
startpoint = zeros(numTracklets,2);
endpoint = zeros(numTracklets,2);


for ind = 1:numTracklets
    
    
        intervals(ind,:) = [centersWorld{ind}(1,3), centersWorld{ind}(end,3)];
        startpoint(ind,:) = [centersWorld{ind}(1,1), centersWorld{ind}(1,2)];
        endpoint(ind,:) = [centersWorld{ind}(end,1), centersWorld{ind}(end,2)];
        
        duration(ind) = centersWorld{ind}(end,3)-centersWorld{ind}(1,3);
        direction = [endpoint(ind,:) - startpoint(ind,:)];
        velocity(ind,:) = direction./duration(ind);
        
    
end








