restoredefaultpath
clear; 
close all; 
opts = get_opts();
opts.sequence=5;
opts.reader = [];

parfor iCam = 1:8
    gen_det_function(opts,iCam);
end