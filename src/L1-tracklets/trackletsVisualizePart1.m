%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHOW DETECTIONS IN WINDOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h = figure(2);
clf('reset');
if opts.dataset ~=2
    imshow(opts.reader.getFrame(opts.current_camera,startFrame));
else
    imshow(opts.reader.getFrame(opts.current_scene, opts.current_camera,startFrame));
end
pause(1);
hold on;
scatter(detectionCenters(:,1),detectionCenters(:,2),[],spatialGroupIDs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%