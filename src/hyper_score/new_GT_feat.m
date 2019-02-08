clear
clc

opts=get_opts();
opts.feature_dir = 'gt_features_ide_basis_train_1fps';

features=[];
for iCam = 1:8
    tmp = h5read(sprintf('%s/L0-features/%s/features%d.h5',opts.dataset_path,opts.feature_dir,iCam),'/emb')';
    tmp(:,3) = local2global(opts.start_frames(iCam), tmp(:,3));
    features   = [features;tmp];
end

hdf5write(sprintf('%s/L0-features/%s/features.h5',opts.dataset_path,opts.feature_dir),'/emb',features');