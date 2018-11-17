clc
clear
opts = get_opts();
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','1fps_train_IDE_40','hyperGT_L2_train_75.h5'),'/hyperGT');
% iCam, pid, centerFrame, SpaGrpID, pos*2, v*2, 0, 256-dim feat
considered_line = all_data(2,:)~=-1;
all_pids = all_data(2,considered_line)';
all_cams = all_data(1,considered_line)';
spagrp_ids = all_data(4,considered_line)';
all_feats = all_data(10:end,considered_line)';

interest_pid = 71;
pos_dists = [];
neg_dists = [];
cams = ones(10,1)*all_cams(all_pids==interest_pid)';
feats = all_feats(all_pids==interest_pid,:)';

feats = mat2gray(feats);
feats = feats(mean(feats,2)>0.1,:);
cams = mat2gray(cams).*ones(1,1,3);
feats = feats.*reshape([51,204,51],1,1,3)/255;
feats = imadjust(feats,[],[],1.5);
img = [cams;feats];
x=3;y=10;
new_img = zeros(size(img,1),75*y,3);
for i=1:75
    new_img(:,(i-1)*y+1:i*y,:) = repmat(img(:,i,:),1,y,1);
end
img = new_img;
new_img = zeros(size(img,1)*x,75*y,3);
for i=1:size(img,1)
    new_img((i-1)*x+1:i*x,:,:) = repmat(img(i,:,:),x,1,1);
end

imwrite(new_img,'temporal locality.png')
imshow(new_img)

