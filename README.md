# DeepCC (Under Construction)
**People Tracking and Re-Identification from Multiple Cameras**

_Ergys Ristani_

[[`Thesis PDF`](http://vision.cs.duke.edu/DukeMTMC/data/misc/Ristani_dissertation.pdf)] [[`DukeMTMC Project Page`](http://vision.cs.duke.edu/DukeMTMC)] [[`BibTeX`](#Citing)]

---
<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/splash.gif" width="900px" />
</div>

Multi-Target Multi-Camera Tracking (MTMCT) is the problem of determining who is where at all times given as input a set of video streams. The output is a set of person trajectories. Person re-identification (ReID) is the problem of retrieving the most similar observation to a query from a gallery of observations, with the goal that the query identity matches the identity of the top rank. A correct match is always assumed to be found in the gallery.


In this repository, we provide MATLAB code to run and evaluate our tracker, as well as Keras code to learn appearance features with a weighted triplet loss. This code has been written over the past years as part of my PhD research, initially for multi-target tracking by correlation clustering (BIPCC), and lately extended to use deep features in multi-camera settings (DeepCC). We additionally provide tools to download and interact with the DukeMTMC dataset. 

---

## Downloading the data

### DukeMTMC 

After cloning this repository you need to download the DukeMTMC dataset. Specify a folder of your choice in `downloadDukeMTMC.m` and run the relevant parts of the script, ommiting the cells which are tagged optional. For the tracker to run you only need to download videos, OpenPose detections, and precomputed detection features. You don't need to extract the frames (~1 TB) because we provide a method to read images directly from videos. 

Please be patient as you are downloading ~160 GB of data. 


### Appearance model

A pre-trained Keras appearance model based on ResNet50 can be downloaded here: [[`Weights`](http://vision.cs.duke.edu/DukeMTMC/data/misc/matlab-weights.hdf5)]
It is required only for across-camera association and should be placed under `experiments/demo/models`. If you didn't know, Matlab is now able to import Keras models.
For single-camera tracking the precomputed features for each detection are used. You can substitute them with your own.

---

## Running the tracker

As a first step you need to set up the dataset root directory. Edit the following line in `get_opts.m`:

```
opts.dataset_path = 'F:/datasets/DukeMTMC/';

```
### Solvers

The graph solver is set in `opts.optimization`. By default Correlation Clustering by a Binary Integer Program (`'BIP`') is used. It solves every graph instance optimally by relying on the Gurobi solver, for which an academic license may be optained for free. 

```
opts.optimization = 'BIP'; 
opts.gurobi_path = 'C:/gurobi800/win64/matlab';

```

If you don't want to use Gurobi, we also provide two existing approximate solvers: Adaptive Label Iterative Conditional Models (`'AL-ICM'`) and Kernighan-Lin (`'KL'`). From our experience, the best trade-off between accuracy and speed is achieved with option `'KL'`.

### Compiling

Run `compile` to obtain mex files for the solvers and helper functions.

### Running

Run `demo` and you will see output logs while the tracker is running. When the tracker completes, you will see the evaluation results for the sequence `trainval-mini`.

## Training an appearance model

In folder `triplet-reid` we provide code for training a ResNet50 CNN with a weighted triplet loss and hard negative mining. This code is heavily based on the [`Hard Triplet Loss code`](https://github.com/VisualComputingInstitute/triplet-reid/). 

Simply run `sh train.sh` to train a model. After training completes, run `sh convert2matlab.sh`. This script converts the Keras model to be imported in Matlab.

## State of the art

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/mchard.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/mceasy.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/schard.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/sceasy.png" width="200px" />
</div>

The state of the art for DukeMTMC is available on [`MOTChallenge`](https://motchallenge.net/results/DukeMTMCT/). Submission instructions can be found on this [`page`](http://vision.cs.duke.edu/DukeMTMC/details.html#evaluation). 

## Discussion

MTMCT and ReID share many similarities because both problems rely on appearance and space-time information. The two problems are _different_ and seem to require different loss functions. In MTMCT the decisions made by the tracker are hard: Two person images either have the same identity or not. In ReID the decisions are soft: The gallery images are ranked without making hard decisions. MTMCT requires a loss that correctly classifies all pairs of observations. ReID instead only requires to a pair of images correctly for any  I query. I will illustrate two ideal feature spaces, one for ReID and one for MTMCT, and argue that the MTMCT classification condition is stronger. 

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/classification.png" width="400px" />
</div>


In MTMCT the ideal feature space should satisfy the classification condition globally, meaning that the largest class variance among _all_ identities should be smaller that the smallest separation margin between _any_ pairs of identities. When this condition holds, a threshold (the maximum class variance) can be found to correctly classify any pair of features as positives or negatives. The classification condition also _implies_ correct ranking in ReID for any given query.

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/ranking.png" width="400px" />
</div>

For correct ranking in ReID it is sufficient that for any query the positive examples are ranked higher than all the negative examples. In the above example the ranking condition is satisfied and guarantees corrent ReID ranking for any query. Yet there exists no threshold that correctly classifies all pairs. Therefore the ReID the ranking condition is _weaker_ than the classification condition. 



## <a name="Citing"></a> Citing



If this code helps your research, please cite the following work which made it possible.

```
@phdthesis{ristani2018thesis,
  author       = {Ergys Ristani}, 
  title        = {People Tracking and Re-Identification from Multiple Cameras},
  school       = {Duke University},
  year         = {2018}
}

@inproceedings{ristani2018features,
  title =        {Features for Multi-Target Multi-Camera Tracking and Re-Identification},
  author =       {Ristani, Ergys and Tomasi, Carlo},
  booktitle =    {Conference on Computer Vision and Pattern Recognition},
  year =         {2018}
}

@inproceedings{ristani2016MTMC,
  title =        {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author =       {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle =    {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year =         {2016}
}

@inproceedings{ristani2014tracking,
  title =        {Tracking Multiple People Online and in Real Time},
  author =       {Ristani, Ergys and Tomasi, Carlo},
  booktitle =    {Asian Conference on Computer Vision},
  year =         {2014},
  pages =        {444--459},
  organization = {Springer}
}
```








