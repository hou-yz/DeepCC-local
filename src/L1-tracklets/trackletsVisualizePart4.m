%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW GENERATED TRACKLETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3);
clf('reset')
imshow(opts.reader.getFrame(opts.current_camera,startFrame));
hold on;

for k = 1:length(ids)
    if isempty(smoothedTracklets), break; end
    
    data = smoothedTracklets(k).data;
    centers = getBoundingBoxCenters(data(:, 3 : 6));
    
    scatter(centers(:,1),centers(:,2),'filled');
    hold on;
end
hold off;
drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%