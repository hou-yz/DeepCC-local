clc
clear
%% Options
opts = get_opts();
opts.sequence = 7;
opts.net.experiment_root = 'experiments/pcb_basis_fc64_train_1fps'; % 'experiments/ide_basis_train_1fps';%
unified_model=~( contains(opts.net.experiment_root,'_icam') || contains(opts.net.experiment_root,'_base_on_uni'));
type='mid' %'1x'%

if unified_model
[thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type);
else
thres_uni=0;diff_p_uni=0;diff_n_uni=0;
end
% [threshold_s,diff_p_s,diff_n_s]=view_distance_distribution_separate_icam(opts,type,unified_model,thres_uni,diff_p_uni,diff_n_uni);