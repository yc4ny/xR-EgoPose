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

### ⚠️ Important Dataset Note

When downloading the dataset from Facebook Research’s official repository
, the data sequence female_001_a_a is incomplete. Some files are missing, which causes mismatches between the number of files in different subfolders. For correct training, the counts of files in these subfolders must always match:

- `rgba`  
- `rot`  
- `worldp`  
- `objectId`  
- `json`  
- `depth`  

If the counts are not equal, training will fail.

**Fix:**  
- The simplest solution is to **remove the entire `female_001_a_a` folder** from your dataset.  
- Alternatively, manually prune the data so that each of the above folders has the **same number of files**, ensuring consistent pairs across all modalities.


## Environment Setup 

Install Conda Environment 
 ```
  conda create -n venv_xrego python=3.9
  conda activate venv_xrego
 ```
Install Pytorch Version 1.7.1
 ```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
 ```

Install Required Packages
 ```
 pip install -r requirements.txt
 ```

## Training - 2D Heatmap Module

![img](doc/architecture_2d.jpg)

To train the 2D Heatmap Estimation Module based on Resnet101 Architecture, run:
```
python train.py --training_type train2d --gpu {gpu id} --log_dir {experiments/Train2d} 
```

The pretrained checkpoint is in `experiments/Train2D/`.

## Training - 3D Pose Lifting Module

![img](doc/architecture_3d.jpg)
To train the 3D Lifting Module, run:
```
python train.py --training_type train3d --gpu {gpu id} --log_dir {experiments/Train3d}
```

The pretrained checkpoint is in `experiments/Train3D/`.

## Finetuning 2D Heatmap & 3D Lifting Module

To finetune the 2D Heatmap and 3D Lifting Module into a single model, run: 
```
python train.py --training_type finetune --gpu {gpu id} --log_dir {experiments/Finetune} --load_2d_model {path to trained 2D Heatmap Module} --load_3d_model {path to trained 3D Pose Lifting Module}
```

The pretrained checkpoint is in `experiments/Finetune/`.

## Evaluation: Demo on the Test Dataset
![img](doc/demo_vis.jpg)
In order to qualitatively and quantitavely evaluate the performance of the model, run the demo:
```
python demo.py --gpu {gpu id} --load_model {path to trained finetuned model} --data{type of data to test on: train, test, val} --save_dir {path of output folder of visualizations}
```

Make sure to load the finetuned model as model path. The default data is set to the testing set. 
The visualization includes the original image with the overlay of predicted 3d joints in orange and the ground truth 3d joints in blue. The terminal prints the MPJPE error in mm. 

## Running Inference on your Custom Data

To run inference on your custom data or other datasets, run:
```
python inference.py --gpu {gpu id} --load_model {path to trained finetuned model} --input_dir {path to folder containing images} --save_dir {path of output folder of visualizations}
```

Since the model is fully trained solely on the xR-EgoPose synthetic dataset, results may not be as expected due to the domain gap. 

## References & Citation
```
@inproceedings{tome2019xr,
  title={xR-EgoPose: Egocentric 3D Human Pose from an HMD Camera},
  author={Tome, Denis and Peluse, Patrick and Agapito, Lourdes and Badino, Hernan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={7728--7738},
  year={2019}
}
```

Base code is adapted from xR-Egopose's official repository.
The base Resnet code is brought from Microsoft Research Bin Xiao (Bin.Xiao@microsoft.com).
Some parts of the implementation, ideas are referenced and adapted from these users: 
@twice154, @ahuard0, @FloralZhao, @jiangyutong. Thanks!
