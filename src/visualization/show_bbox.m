function fig1=show_bbox(opts,iCam,frame,bbox)
img = opts.reader.getFrame(iCam,frame);
fig1 = img(bbox(2):min(bbox(2)+bbox(4),size(img,1)),bbox(1):min(bbox(1)+bbox(3),size(img,2)),:);
% imshow(fig1)
end

