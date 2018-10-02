clc
clear
%% Options
opts = get_opts();
opts.net.experiment_root =  'experiments/fc256_1fps';%'experiments/fc256_30fps_separate_icam_camstyle_64_23'; %
unified_model=~contains(opts.net.experiment_root,'_icam');
type='mid' %'1x'%

if unified_model
[thres_uni,diff_p_uni,diff_n_uni]=view_distance_distribution(opts,type);
else
thres_uni=0;diff_p_uni=0;diff_n_uni=0;
end
[threshold_s,diff_p_s,diff_n_s]=view_distance_distribution_separate_icam(opts,type,unified_model,thres_uni,diff_p_uni,diff_n_uni);