function snapshot = get_bb( img, bb )
%SHOW_BB Summary of this function goes here
%   Detailed explanation goes here
bb = round(bb);
snapshot = img(max(1,bb(2)):min(1080,bb(2)+bb(4)),max(1,bb(1)):min(1920,bb(1)+bb(3)),:);

end

