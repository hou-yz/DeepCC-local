restoredefaultpath
clear; 
close all; 
opts = get_opts();
opts.sequence=2;
opts.reader = [];

for iCam = 1:8
    gen_det_function(opts,iCam);
end