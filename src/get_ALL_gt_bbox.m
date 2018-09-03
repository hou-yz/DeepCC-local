restoredefaultpath
clear; 
close all; 
opts = get_opts();
opts.sequence=1;
opts.reader = [];

for iCam = 1:8
    gen_gt_function(opts,iCam);
end