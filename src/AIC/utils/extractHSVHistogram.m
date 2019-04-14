function [ feature ] = extractHSVHistogram( image )

binsH = 16;
binsS = 16;
binsV = 4;

img = rgb2hsv(image); 

h = img(:,:,1);
s = img(:,:,2);
v = img(:,:,3);
    
histH = hist(h(:),binsH);
histS = hist(s(:),binsS);
histV = hist(v(:),binsV);

feature = [histH, histS, histV];
feature = feature/(eps+sum(feature));