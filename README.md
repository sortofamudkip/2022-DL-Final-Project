# 2022-DL-Final-Project
The file contains an outline of the project as well as its contents.

Members:
* Jonathan Cedeno
* Elena Golimblevskaia
* Youssef Mecky
* Wishyut Pitawanik
* Niklas Riekenbrauck


## Links
* The Colab demo file can be found [here](https://colab.research.google.com/drive/1S5pJnkNnFQOg1wkDW5q-kJhHEHNOMR9C?usp=sharing).
* The link to the Github repository can be found **THIS NEEDS TO BE ADDED**.

### Python Setup
Popular Python environments such as conda, virtualenv can be used. First, obtain the dependencies from `requirements.txt`:

`pip install -r requirements.txt`

### Setup Kaggle for data loading
In order to download the dataset, Kaggle is requried. The following steps should be taken:
1. Create account on [Kaggle](https://www.kaggle.com/)
2. Go to `https://www.kaggle.com/<ACCOUNT_NAME>/account`
2. Click "Create New API Token", which downloads a token file
4. Move the downloaded token file to `~/.kaggle/kaggle.json`

Calling `./data_loading.py` will cause the the dataset will be downloaded into the `data/` folder in this repository. When training or testing for the first time, the data will also be downloaded, if it hasn't been already.

### Setup Wandb
Weights and Biases (wandb) is throughout the training and testing process to store model performance metrics.

To run the code in this repository, a wandb account must be created, as described [here](https://docs.wandb.ai/quickstart). When training or testing a model, it is recommended to add relevant tags in the CLI arguments in order to organize the results.

## Colab Demo
**(explain the demo here)**

## Quickstart
For this section, the root directory is the directory of the README (i.e., the top-level directory for this repo).

### Download data
To download the dataset, ensure you have the correct Kaggle API credentials and then run `python ./data_loading.py`. 
The 6.31GB data should then be downloaded to the folder `./data/`.

### Train models
After downloading the data, the `./train.py` file can then be used to train a model. 

* Train demo model (**DEMO**, not used in final report): `python train.py --data_path="data" --num_epochs=5 --model=demo --model_state_file="model_demo.pt"` 
* Train ResNet-18 trained for 5 epochs (**RES5**): `python train.py --data_path="data" --num_epochs=5 --model=resnet --model_state_file="model_resnet.pt"` 
* Train ResNet-18 with data augmentation for 5 epochs (**RESAUG5**): `python train.py --data_path="data" --num_epochs=5 --model=resnet_augmented --model_state_file="model_resnet_aug.pt"` 
* Train RESAUG5 trained for 5 more epochs (**RESAUG10**): `python train.py --data_path="data" --num_epochs=5 --model=resnet_augmented --model_state_file="RES5_MODEL_NAME" --resume_training` 
* Train VGG-16 for 5 epochs (**VGG5**): ` python train.py --data_path="data" --num_epochs=5 --model=vgg16_pretrained --model_state_file="model_vgg.pt"` 


### Test models
After the models are trained and saved, the `./test.py` file can then be used to test a model. 

* Test demo model (not used in final report): `python test.py --data_path="data" --model=demo --model_state_file="DEMO_MODEL_NAME"` 
* Test ResNet-18 trained for 5 epochs (**RES5**): `python test.py --data_path="data" --model=resnet --model_state_file="RES5_MODEL_NAME"` 
* Test ResNet-18 with data augmentation, trained for 5 epochs (**RESAUG5**): `python test.py --data_path="data" --model=resnet_augmented --model_state_file="RESAUG5_MODEL_NAME"` 
* Test RESAUG5 trained for 5 more epochs (**RESAUG10**): `python test.py --data_path="data" --model=resnet_augmented --model_state_file="RESAUG10_MODEL_NAME"` 
* Test VGG-16 (**VGG5**): `python test.py --data_path="data" --model=vgg16_pretrained --model_state_file="VGG5_MODEL_NAME"` 

## Saved models
The saved trained models can be found in the following links:
* [RES5](https://drive.google.com/file/d/1u0heklzsMb65usgu9DeNgk90nEphV4K4/view?usp=sharing)
* [RESAUG5](https://drive.google.com/file/d/12h7zSOUw1FqdMRfRYhn8RIKQiysbjyeo/view?usp=sharing)
* [RESAUG10](https://drive.google.com/file/d/1IsBM3rn0q23qyoHgOckXi5QoAi3w-ccn/view?usp=sharing)
* [VGG5](https://drive.google.com/file/d/1-ePhNFry-z_f5DNbxCxBN9BDMFxRb2a6/view?usp=sharing)

## Dataset link
The link to the dataset can be found [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data).

## Resources used
* I don't know what this refers to
