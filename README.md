# DeepCC
**Features for Multi-Target Multi-Camera Tracking and Re-Identification. CVPR 2018**

_Ergys Ristani, Carlo Tomasi_

[[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)] [[Spotlight](https://www.youtube.com/watch?v=GBo4sFNzhtU&feature=youtu.be&t=5462)] [[PhD Thesis](http://vision.cs.duke.edu/DukeMTMC/data/misc/Ristani_dissertation.pdf)] [[PhD Slides](http://vision.cs.duke.edu/DukeMTMC/data/misc/Ristani_slides.pdf)] [[DukeMTMC Project](http://vision.cs.duke.edu/DukeMTMC)] [[BibTeX](#Citing)]

---
<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/splash.gif?maxAge=2592000" width="900px" />
</div>

Multi-Target Multi-Camera Tracking (MTMCT) is the problem of determining who is where at all times given a set of video streams as input. The output is a set of person trajectories. Person re-identification (ReID) is a closely related problem. Given a query image of a person, the goal is to retrieve from a database of images taken by different cameras the images where the same person
appears. 


In this repository, we provide MATLAB code to run and evaluate our tracker, as well as Tensorflow code to learn appearance features with our weighted triplet loss. This code has been written over the past years as part of my PhD research, initially for multi-target tracking by correlation clustering (BIPCC), and lately extended to use deep features in multi-camera settings (DeepCC). We additionally provide tools to download and interact with the DukeMTMC dataset. 

---

## Downloading the data

### DukeMTMC 

After cloning this repository you need to download the DukeMTMC dataset. Specify a folder of your choice in `src/duke/downloadDukeMTMC.m` and run the relevant parts of the script, omitting the cells which are tagged optional. For the tracker to run you only need to download videos, OpenPose detections, and precomputed detection features. 

Please be patient as you are downloading ~160 GB of data. [[`md5sum`](http://vision.cs.duke.edu/DukeMTMC/data/videos/md5sum.txt)]

---

## Running the tracker

As a first step you need to set up the dataset root directory. Edit the following line in `get_opts.m`:

```
opts.dataset_path = 'F:/datasets/DukeMTMC/';
```
### Dependencies

Clone [mexopencv](https://github.com/kyamagu/mexopencv) in `src/external/` and follow its installation instructions. This interface is used to read images directly from the Duke videos.

### Pre-computed features

Download the [pre-computed features](http://vision.cs.duke.edu/DukeMTMC/data/detections/openpose/features/) into  `experiments/demo/L0-features/`.


### Compiling

Run `compile` to obtain mex files for the solvers and helper functions.

### Training an appearance model

To train and evaluate our appearance model which employs the weighted triplet loss, first download [resnet_v1_50.ckpt](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) in `src/triplet-reid/`. Then install [imgaug](https://github.com/aleju/imgaug). Finally run:
```
opts = get_opts();
train_duke(opts);
embed(opts);
evaluation_res_duke_fast(opts);
```
The code will run 25,000 training iterations, compute embeddings for query and gallery images of the DukeMTMC-reID benchmark, and finally print the mAP and rank-1 score. The above functions are MATLAB interfaces to the Tensorflow/Python code of [Beyer et al.](https://github.com/VisualComputingInstitute/triplet-reid/) The code has been extended to include our weighted triplet loss.

Alternatively you can run `train_duke_hnm` to train with hard negative mining.

Once you train a model, you can analyze the distribution of distances between features to obtain a separation threshold:
```
view_distance_distribution(opts);
```

You can also use `features = embed_detections(opts, detections);` to compute features for a set of detections in the format [camera, frame, left, top, width, height];

### Running DeepCC

Run `demo` and you will see output logs while the tracker is running. When the tracker completes, you will see the quantitative evaluation results for the sequence `trainval-mini`.

### Understanding errors

To gain qualitative insights why the tracker fails you can run `render_results(opts)`

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/understanding_errors.jpg?maxAge=2592000" width="500px" />
</div>

This will generate movies with the rendered trajectories validated against ground truth using the ID measures. Color-coded tails with IDTP, IDFP and IDFN give an intuition for the tracker's failures. The movies will be placed under `experiments/demo/video-results`.

### Note on solvers

The graph solver is set in `opts.optimization`. By default Correlation Clustering by a Binary Integer Program (`BIPCC`) is used. It solves every graph instance optimally by relying on the Gurobi solver, for which an academic license may be obtained for free. 

```
opts.optimization = 'BIPCC'; 
opts.gurobi_path = 'C:/gurobi800/win64/matlab';
```

If you don't want to use Gurobi, we also provide two existing approximate solvers: Adaptive Label Iterative Conditional Models (`AL-ICM`) and Kernighan-Lin (`KL`). From our experience, the best trade-off between accuracy and speed is achieved with option `'KL'`.

## Visualization

To visualize the detections you can run the demo `show_detections`.

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/pose_detections.jpg?maxAge=2592000" width="500px" />
</div>


You can run `render_trajectories_top` or `render_trajectories_side` to generate a video animation similar to the gif playing at the top of this page.

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/top_view.jpg?maxAge=2592000" height="150px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/side_view.jpg?maxAge=2592000" height="150px" />
</div>

To generate ID Precision/Recall plots like in the state of the art section see `render_state_of_the_art`. Make sure that you update the files provided in `src/visualization/data/duke_*_scores.txt` with the latest MOTChallenge submissions. The provided scores are only supplied as a reference. 


## State of the art

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/mchard.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/mceasy.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/schard.png" width="200px" />
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/sceasy.png" width="200px" />
</div>

The state of the art for DukeMTMC is available on [`MOTChallenge`](https://motchallenge.net/results/DukeMTMCT/). Submission instructions can be found on this [`page`](http://vision.cs.duke.edu/DukeMTMC/details.html#evaluation). 

The paper's submission file `duke.txt` can be downloaded [here](http://vision.cs.duke.edu/DukeMTMC/data/misc/DeepCC.zip). Results from the released tracker may differ from the original one due changes in code and settings. Once you are happy with the performance of your extensions to DeepCC, run `prepareMOTChallengeSubmission(opts)` to obtain a submission file `duke.txt` for MOTChallenge. 

## Remarks

MTMCT and ReID problems differ subtly but _fundamentally_. In MTMCT the decisions made by the tracker are hard: Two person images either have the same identity or not. In ReID the decisions are soft: The gallery images are ranked without making hard decisions. MTMCT training requires a loss that correctly _classifies_ all pairs of observations. ReID instead only requires a loss that correctly _ranks_ a pair of images by which is most similar to the query. Below I illustrate two ideal feature spaces, one for ReID and one for MTMCT, and argue that the MTMCT classification condition is stronger. 

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/classification.png" width="400px" />
</div>


In MTMCT the ideal feature space should satisfy the classification condition globally, meaning that the largest class variance among _all_ identities should be smaller than the smallest separation margin between _any_ pair of identities. When this condition holds, a threshold (the maximum class variance) can be found to correctly classify any pair of features as co-identical or not. The classification condition also _implies_ correct ranking in ReID for any given query. 

<div align="center">
  <img src="http://vision.cs.duke.edu/DukeMTMC/img/ranking.png" width="500px" />
</div>

For correct ranking in ReID it is sufficient that for any query the positive examples are ranked higher than all the negative examples. In the above example the ranking condition is satisfied and guarantees correct ReID ranking for any query. Yet there exists no threshold that correctly classifies all pairs. Therefore the ReID ranking condition is subsumed by the MTMCT classification condition. 



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








