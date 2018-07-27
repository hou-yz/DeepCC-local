===============================================
Large Scale Correlation Clustering Optimization
===============================================

1. Included packages
--------------------
This distribution includes the following package
QPBO - see below

2. Main functions
-----------------
This package contains the following functions
a_expand.m - implementation fo the expand&explore algorithm (Alg. 1 in Bagon&Galun2011)
ab_swap.m  - implementation fo the swap&explore algorithm (Alg. 2 in Bagon&Galun2011)
AL_ICM.m   - implementation fo the adaptive label ICM (Sec. 5.2 in Bagon&Galun2011)

3. Installation
---------------
a. Download the tarball from the website and extract it to a local folder.
b. Make sure your mex compiler under Matlab is well setup.
   If it does not, type (in Matlab)
   >> mex -setup
   and chhose either the visual studio compiler (for PC), or the gcc compiler (of Linux)
c. Change directory in Matlab to the local folder in which you extracted the tarball
d. Run:
   >> mexall
   This will compile several mex files required by our algorithms.
e. If no error appear - you are good to go...

4. Documentation
----------------
Use (in Matlab)
>> doc <function name>
To see usage instructions and documentation for the various functions.

5. Usage example
----------------
Here is a small usage example (you may use this example to test your installation)
You may copy-paste these commands into Matlab:
%
[gt w] = MakeSynthAff(100, [1 3 5 2], 10, .5, .1); % creates a synthetic sparse affinity matrix
plotWl(w,gt);     % plot the matrix ordered according to the ground-truth labeling
el = a_expand(w); % CC optimization using expand&explore
plotWl(w,el);     % visualize the result
sl = ab_swap(w);  % CC optimization using swap&explore
plotWl(w,sl);     % visualize the result
il = AL_ICM(w);   % CC optimization using adaptive-label ICM
plotWl(w,il);     % visualize the result
[CCEnergy(w,gt) CCEnergy(w,el) CCEnergy(w,sl) CCEnergy(w,il)], % output CC objective values for the different partitions.

6. Proper reference
-------------------
Using this software in any academic work you must cite the following works in any resulting publication:
S. Bagon and M. Galun. "Large Scale Correlation Clustering Optimization", arXiv'2011
C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer. "Optimizing binary MRFs via extended roof duality", CVPR'2007


A. Included package:
QPBO: Vladimir Kolmogorov's implementation
------------------------------------------
source files from:
http://pub.ist.ac.at/~vnk/software.html
