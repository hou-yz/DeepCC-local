clc
clear
%% Options
mot=1;
if mot
    opts = get_opts_mot();
    opts.feature_dir = 'D:/Data/MOT16/gt_feat/';
    opts.net.experiment_root = 'og512';%
else
    opts = get_opts();
    opts.sequence = 7;
    opts.net.experiment_root = 'experiments/ide_basis_train_1fps';%
end
type='mid' %'1x'%

[thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type,mot);
