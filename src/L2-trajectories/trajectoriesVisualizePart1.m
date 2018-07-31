%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VISUALIZE 1: SHOW ALL TRACKLETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% backgroundImage =  imfuse(readFrame(dataset, max(1,startTime)),readFrame(dataset,min(endTime,dataset.endingFrame+syncTimeAcrossCameras(dataset.camera))),'blend','Scaling','joint');
startImg = opts.reader.getFrame(opts.current_camera, max(1,startTime));
endImg = opts.reader.getFrame(opts.current_camera, max(1,endTime));
backgroundImage =  imfuse(startImg, endImg,'blend','Scaling','joint');

imshow(backgroundImage);

hold on;

trCount = 0;
for k = 1 : length(currentTrajectories)
    
    for i = 1:length(currentTrajectories(k).tracklets)
        trCount = trCount +1;
        detections = currentTrajectories(k).tracklets(i).data;
        trackletCentersView = getBoundingBoxCenters(detections(:,[3:6]));
        scatter(trackletCentersView(:,1),trackletCentersView(:,2),'filled');
        total = size(trackletCentersView,1);
        text(trackletCentersView(round(total/2),1),trackletCentersView(round(total/2),2)+0.01,sprintf('(%d,%d),I:%d,F:%d)',k,i,trCount,min(detections(:,1))));
        hold on;
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

