# *x*R-EgoPose

The *x*R-EgoPose Dataset has been introduced in the paper ["*x*R-EgoPose: Egocentric 3D Human Pose from an HMD Camera"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tome_xR-EgoPose_Egocentric_3D_Human_Pose_From_an_HMD_Camera_ICCV_2019_paper.pdf) (ICCV 2019, oral). It is a dataset of ~380 thousand photo-realistic *egocentric*  camera images in a variety of indoor and  outdoor spaces.

![img](doc/teaser.jpg)


The code contained in this repository is a PyTorch implementation of training and inference code of the model. 

## Downloading the *x*R-EgoPose Dataset

Download the official data set from the [official repository](https://github.com/facebookresearch/xR-EgoPose/releases/tag/v1.0).
The authors provided the downloaded script but it seems to be broken and does not download all the files. It is recommended to download the zip files from the provided link manually.

Once you have downloaded all the tar.gz files, run 
```
python utils/extract_data.py --input {path of the downloaded tar.gz zip files} --output {path to extracted files}
```

Please create folders:

```
/TrainSet
/ValSet
/TestSet
```
Then, put the extracted output folders according to the set type as shown below. 

|Train-set| Test-set | Val-set |
|---------|----------|---------|
|female_001_a_a |female_004_a_a | male_008_a_a |
|female_002_a_a |female_008_a_a | |
|female_002_f_s |female_010_a_a | |
|female_003_a_a |female_012_a_a | |
|female_005_a_a |female_012_f_s | |
|female_006_a_a |male_001_a_a | |
|female_007_a_a |male_002_a_a | |
|female_009_a_a |male_004_f_s | |
|female_011_a_a |male_006_a_a | |
|female_014_a_a |male_007_f_s | |
|female_015_a_a |male_010_a_a | |
|male_003_f_s |male_014_f_s | |
|male_004_a_a | | |
|male_005_a_a | | |
|male_006_f_s | | |
|male_007_a_a | | |
|male_008_f_s | | |
|male_009_a_a | | |
|male_010_f_s | | |
|male_011_f_s | | |
|male_014_a_a | | |

The organized folder structure of dataset should look something like this: 

```
TrainSet
├── female_001_a_a
│   ├── env 01
│   │   └── cam_down
│   │   	├── depth
│   │   	├── json
│   │   	├── objectId
│   │   	├── rgba
│   │   	├── rot
│   │   	└── worldp
│   ├── ...
│   └── env 03
│ 
ValSet
│  
│  
TestSet 
```


## Training - 2D Heatmap Module

![img](doc/architecture_2d.png)

To train the 2D Heatmap joint estimation module, run
```
python train.py --training_type train2d --gpu {gpu id} --log_dir {experiments/Train2d} --load_model {for resuming training from checkpoint}
```

You can also download the pretrained checkpoint from this [link](https://drive.google.com/drive/folders/1vAmK83MO3UvVd52OQ3X6G8gPQhClZHL0?usp=sharing). The checkpoint is located under '''Train2d''' folder. 