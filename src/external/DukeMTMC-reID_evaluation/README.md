## DukeMTMC-reID Description
![](https://github.com/layumi/Duke_evaluation/blob/master/DukeMTMC-reID_mosaic.jpg)

***What's new:*** Following the license on the [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/) website, we added a few modifications to the license terms. You may check the license in this repo. The dataset is released only for academic research. 

[DukeMTMC-reID](#dukemtmc-reid-description) [1] is a subset of the [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/) dataset [2] for image-based re-identification, in the format of the [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset. The original dataset contains 85-minute high-resolution videos from 8 different cameras. Hand-drawn pedestrain bounding boxes are available. 

We crop pedestrain images from the videos every 120 frames, yielding in total 36,411 bounding boxes with IDs. There are 1,404 identities appearing in more than two cameras and 408 identities (distractor ID) who appear in only one camera. We randomly select 702 IDs as the training set and the remaining 702 IDs as the testing set. In the testing set, we pick one query image for each ID in each camera and put the remaining images in the gallery. 

**As a result, we get 16,522 training images of 702 identities, 2,228 query images of the other 702 identities and 17,661 gallery images (702 ID + 408 distractor ID).** 

## Table of contents
* [DukeMTMC-reID](#dukemtmc-reid-description)
    * [About Dataset](#about-dataset)
    * [Dataset Licence](#dataset-licence)
    * [Download Dataset](#download-dataset)
    * [Dataset Insights](#dataset-insights)
* [Evaluation](#evaluation)
    * [Evaluation](#evaluation)
    * [State-of-the-art Methods](#state-of-the-art)
    * [Baseline](#baseline)
* [Related Datasets](#dukemtmc-attribute)
    * [DukeMTMC-attribute](#dukemtmc-attribute)
    * [DukeMTMC-Pose](#dukemtmc-pose)
* [Reference](#reference)

### About Dataset
|File  | Description | 
| --------   | -----  |
|/bounding_box_test  | The gallery images. We retrieve a query from this image pool.|
|/bounding_box_train  | The training images. This dir contains the images from 702 different identities.|
|/query  | The query images. Each of them is from different identities in different cameras.|

**Naming Rule of the images** In bbox "0005_c2_f0046985.jpg", "0005" is the identity. "c2" means the image from Camera 2. "f0046985" is the 46985th frame in the video of Camera 2.

### Dataset Licence
Please follow the [LICENSE_DukeMTMC-reID](https://github.com/layumi/DukeMTMC_reID_evaluation/blob/master/LICENSE_DukeMTMC-reID.txt). You are free to share, create and adapt the DukeMTMC-reID dataset, in the manner specified in the license. 

We also include the [LICENSE_DukeMTMC](https://github.com/layumi/DukeMTMC_reID_evaluation/blob/master/LICENSE_DukeMTMC.txt). If you want to share, create and adapt the DukeMTMC dataset, please follow this license.

The DukeMTMC-reID evaluation code is under the MIT License.

### Download Dataset

The direct download link is [Here](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip).

You also can download the DukeMTMC-reID dataset from [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh).

Some unzip tools on Windows may meet some problems. Please check that you have the following files after unzip:
![](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/file.png)

If download links are unavailable, please don't hesitate to contact me to update links. Thank you.

### Dataset Insights

* Data Distribution
![](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/Data_Distribution.jpg)

Figure. The image distribution of DukeMTMC-reID training set. We note that the median of images per ID is 20. But some ID may contain lots of images, which may compromise some algorithms. (For example, ID 5388 contains 426 images.) 

Thank [Xun](https://github.com/Xun-Yang) for suggestions.

* Camera Topology
![](http://vision.cs.duke.edu/DukeMTMC/img/campus.png)

This picture is from [DukeMTMC Homepage](http://vision.cs.duke.edu/DukeMTMC/).

### Evaluation
(Matlab)To evaluate, you need to calculate your gallery and query feature (i.e., 17661x2048 and 2228x2048 matrix) and save them in advance. Then download the codes in this repository. You just need to change the image path and the feature path in the `evaluation_res_duke_fast.m` and run it to evaluate.

(Python)We also provide an evaluation code in python. You may refer to [here](https://github.com/layumi/Person_reID_baseline_pytorch#evaluation).

### State-of-the-art
We have summarized current state-of-the-art methods on DukeMTMC at [here](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/State-of-the-art/README.md). If you notice any result that has not been included in this table, please connect [Zhedong Zheng](mailto:zdzheng12@gmail.com) without hesitation to add the method. You are welcomed!

### Baseline
We release our baseline training code and pretrained model in [[Matconvnet Version]](https://github.com/layumi/DukeMTMC-reID_baseline) and [[Pytorch Version]](https://github.com/layumi/Person_reID_baseline_pytorch). You can choose one of the two tools to conduct the experiment. Furthermore, you may try our new [Pedestrain Alignment Code](https://github.com/layumi/Pedestrian_Alignment) which combines person alignment with re-ID.

Or you can directly download the finetuned ResNet-50 baseline feature. You can download it from [GoogleDriver](https://drive.google.com/open?id=0B0VOCNYh8HeRVFR6bldBX0lTRVE) or [BaiduYun](https://pan.baidu.com/s/1c2CIsTy), which includes the feature of training set, query set and gallery set. The DukeMTMC-reID LICENSE is also included.

### Sample Retrieval
![](https://github.com/layumi/Duke_evaluation/blob/master/duke_rank.jpg)

### DukeMTMC-attribute
We also annotated 23 human-level attributes (gender/clothing/...) for DukeMTMC-reID. You can find it in the following link:
https://github.com/vana77/DukeMTMC-attribute

![](https://github.com/vana77/DukeMTMC-attribute/blob/master/sample_image.jpg)

### DukeMTMC-Pose
We use pretrained CNN to generate 18 body keypoints. You can find it in the following link:
https://github.com/layumi/DukeMTMC-Pose

![](https://github.com/layumi/DukeMTMC-Pose/blob/master/demo.png) 

### References
- [1] Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. Zheng et al., ICCV 2017

- [2] Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. Ristani et al., ECCVWS 2016

Please cite the following two papers if this dataset helps your research.
```
@inproceedings{zheng2017unlabeled,
  title={Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2017}
}

@inproceedings{ristani2016MTMC,
  title = {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author = {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle = {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year = {2016}
}
```
