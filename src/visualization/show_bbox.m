function show_bbox(opts,iCam,frame,bbox1,bbox2)
img = opts.reader.getFrame(iCam,frame);
fig1 = img(bbox1(2):bbox1(2)+bbox1(4),bbox1(1):bbox1(1)+bbox1(3),:);
subplot(1,2,1)
imshow(fig1)
fig2 = img(bbox2(2):bbox2(2)+bbox2(4),bbox2(1):bbox2(1)+bbox2(3),:);
subplot(1,2,2)
imshow(fig1)
end

