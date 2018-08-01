Multiple Object Tracking Challenge Development Kit
==================================================

[MOTChallenge.net](https://motchallenge.net)

## Authors

	* Anton Milan (anton.milan@adelaide.edu.au)
	* Ergys Ristani (ristani@cs.duke.edu)


Description
===========

Version 1.4

This development kit provides scripts to evaluate detection/tracking results.
Please report bugs to: 

    Anton Milan   - anton.milan@adelaide.edu.au
    Ergys Ristani - ristani@cs.duke.edu


Requirements
============
- MATLAB
- C/C++ compiler
- Benchmark data for MOT15-17
  e.g. 2DMOT2015, available here: http://motchallenge.net/data/2D_MOT_2015/


- Note 1: DukeMTMCT benchmark data will download automatically.
- Note 2: The code has been tested under Windows and Linux.
  

Usage
=====

1) Run: 
       
       compile

2) Run one of the following:

       demo_evalMOT15
       demo_evalMOT15_3D
       demo_evalMOT16
       demo_evalMOT17Det
       demo_evalDukeMTMCT

   Note: For demo_evalMOT1X you need to replace the benchmarkGtDir path to point to the training set data. For example:
         
         benchmarkGtDir = '../data/2DMOT2015/train/';
         allMets = evaluateTracking('c2-train.txt', 'res/MOT15/data/', benchmarkGtDir, 'MOT15');


You should see the following output:

    >> demo_evalMOT15
    Sequences: 
        'TUD-Stadtmitte'
        'TUD-Campus'
        'PETS09-S2L1'
        'ETH-Bahnhof'
        'ETH-Sunnyday'
        'ETH-Pedcross2'
        'ADL-Rundle-6'
        'ADL-Rundle-8'
        'KITTI-13'
        'KITTI-17'
        'Venice-2'

        ... TUD-Stadtmitte
    TUD-Stadtmitte
     IDF1  IDP  IDR| Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM| MOTA  MOTP MOTAL 
     64.5 82.0 53.1| 60.9  94.0  0.25| 10   5   4   1|   45   452    7    6| 56.4  65.4  56.9 

        ... TUD-Campus
    TUD-Campus
     IDF1  IDP  IDR| Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM| MOTA  MOTP MOTAL 
     55.8 73.0 45.1| 58.2  94.1  0.18|  8   1   6   1|   13   150    7    7| 52.6  72.3  54.3 

    ...

        ... Venice-2
    Venice-2
     IDF1  IDP  IDR| Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM| MOTA  MOTP MOTAL 
     35.5 43.6 29.9| 42.0  61.3  3.15| 26   4  16   6| 1890  4144   42   52| 14.9  72.6  15.5 


     ********************* Your 2DMOT15 Results *********************
     IDF1  IDP  IDR| Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM| MOTA  MOTP MOTAL 
     41.2 53.2 33.6| 45.3  71.7  1.30|500  81 161 258| 7129 21842  220  338| 26.8  72.4  27.4 


Details
=======
The tracking evaluation script accepts 4 arguments:

    evaluateTracking(seqmap, resDir, gtDataDir, benchmark)

1) seqmap
sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
evaluated in a single run. These files are inside the ./seqmaps folder.

2) resDir
The folder containing the tracking results. Each one should be saved in a
separate .txt file with the name of the respective sequence (see ./res/data)

3) gtDataDir
The folder containing the ground truth files.

4) benchmark
The name of the benchmark, e.g. 'MOT15', 'MOT16', 'MOT17', 'DukeMTMCT'

The results will be shown for each individual sequence as well as for the
entire benchmark. Benchmark scores are aggregate scores for all sequences.



Directory structure
===================
	

./res
----------
This directory contains 
  - the tracking results for each sequence in a subfolder data  
  - eval.txt, which shows all metrics for this demo
  
  
  
./utils
-------
Various scripts and functions used for evaluation.


./seqmaps
---------
Sequence lists for different benchmarks




Version history
===============

1.4 - Sep 30, 2017
  - Bitbucket release

1.3 - Apr 29, 2017
  - Merged single- and multi-camera evaluation branches
  - Code cleanup
  - Evaluation code ported to C++

1.2 - Apr 16, 2017
  - Included evaluation for detections
  - Made evaluation script more efficient	

1.1.1 - Oct 10, 2016
  - Included camera projections scripts
	
1.1 - Feb 25, 2016
  - Included evaluation for the new MOT16 benchmark

1.0.5 - Nov 10, 2015
  - Fixed bug where result has only one frame
  - Fixed bug where results have extreme values for IDs
  - Results may now contain invalid frames, IDs, which will be ignored

1.0.4 - Oct 08, 2015
  - Fixed bug where result has more frames than ground truth

1.0.3 - Jul 04, 2015
  - Removed spurious frames from ETH-Pedcross2 result (thanks Nikos)
  
1.0.2 - Mar 11, 2015
  - Fix to exclude small bounding boxes from ground truth
  - Special case of empty mapping fixed

1.0.1 - Feb 06, 2015
  - Fixes in 3D evaluation (thanks Michael)

1.00 - Jan 23, 2015
  - initial release