cur_dir = pwd;

% KL
cd src/correlation_clustering/external/KL
mex -O -largeArrayDims Multicut_KL_MEX.cpp
cd(cur_dir)

% AL-ICM
cd src/correlation_clustering/external/AL-ICM
mex -O -largeArrayDims pure_potts_icm_iter_mex.cpp
cd(cur_dir)

% Combinator
cd src/external/combinator
mex -O -largeArrayDims cumsumall.cpp
cd(cur_dir)
