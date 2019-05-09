clc
clear
%% Options
dataset=2;
if dataset ==0
    opts = get_opts();
    opts.sequence = 7;
    opts.net.experiment_root = 'experiments/pcb_basis_fc64_train_1fps';%'experiments/ide_basis_train_1fps';%
elseif dataset == 1
    opts = get_opts_mot();
    opts.feature_dir = 'D:/Data/MOT16/gt_feat/';
    opts.net.experiment_root = 'ide256';%'og512';%
elseif dataset == 2
    opts = get_opts_aic();
    opts.sequence = 1;
    opts.net.experiment_root = 'experiments/zju_lr001_ensemble_gt_trainval';%'experiments/zju_best_labeled_trainval';
end
type='mid' %'1x'%

[thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type,dataset);
