clc
clear

opts = get_opts_aic();
opts.experiment_name = 'aic_label_det';
opts.sequence = 3;

testData = [];

scene = 2;


for iCam = opts.cams_in_scene{scene}
    resdata = dlmread(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.sequence_names{opts.sequence}));
    resdata = round(resdata);
    resdata(:,[1,2]) = resdata(:,[2,1]);
    resdata(:,[7:end]) = [];
    resdata = [ones(size(resdata,1),1)*iCam,resdata];
    testData = [testData;resdata];
end

testData = [testData,ones(size(testData,1),4)*-1];
testData = sortrows(testData,[1,3]);

save(fullfile(opts.dataset_path,'ground_truth', 'test_easy.mat'),'testData');
