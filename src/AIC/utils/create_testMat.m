clc
clear

opts = get_opts_aic();
opts.experiment_name = 'aic_label_det';
opts.sequence = 2;

% removeOverlapping(opts);


% testData = [];
% 
% scene = 6;
% 
% 
% for iCam = opts.cams_in_scene{scene}
%     resdata = dlmread(sprintf('%s/%s/L3-identities/cam%d_%s.txt',opts.experiment_root, opts.experiment_name, iCam,opts.folder_by_seq{opts.sequence}));
%     resdata = round(resdata);
%     resdata(:,[1,2]) = resdata(:,[2,1]);
%     resdata(:,[7:end]) = [];
%     resdata = [ones(size(resdata,1),1)*iCam,resdata];
%     testData = [testData;resdata];
% end
% 
% testData = [testData,ones(size(testData,1),4)*-1];
% testData = sortrows(testData,[1,3]);
s02 = load(fullfile(opts.dataset_path,'ground_truth', 'test_easy.mat'));
max_id = max(s02.testData(:,2));
s05 = load(fullfile(opts.dataset_path,'ground_truth', 'test_mini.mat'));
s05.testData(:,2) = s05.testData(:,2) + max_id;
testData = [s02.testData;s05.testData];

save(fullfile(opts.dataset_path,'ground_truth', 'test_labeled.mat'),'testData');
