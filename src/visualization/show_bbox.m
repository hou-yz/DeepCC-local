function fig1=show_bbox(opts,iCam,frame,bbox)
if opts.dataset ~= 2
    img = opts.reader.getFrame(iCam,frame);
else
    img = opts.reader.getFrame(opts.current_scene, iCam,frame);
end
fig1 = img(bbox(2):min(bbox(2)+bbox(4),size(img,1)),bbox(1):min(bbox(1)+bbox(3),size(img,2)),:);
% imshow(fig1)
end

