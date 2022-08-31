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
* The link to the Github repository can be found [here](https://github.com/sortofamudkip/2022-DL-Final-Project).

### Python Setup
Popular Python environments such as conda, virtualenv can be used. First, obtain the dependencies from `requirements.txt`:

`pip install -r requirements.txt`

We recommend the usage of Python3. Our scripts were tested on Linux and OS X.

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
In this demo we hoped to show a bit of the code used in the experiments and show a proof of concept in some sense that is why we only used 512 images and show the main structutre of our training and evaulation however we can't draw any conclusions here due to the small size of the sample used

## Quickstart
For this section, the root directory is the directory of the README (i.e., the top-level directory for this repo). 


> Our main model should be RESAUG10 , check save models for reference 



### Download data
To download the dataset, ensure you have the correct Kaggle API credentials and then run `python data_loading.py`. 
The 6.31GB data should then be downloaded to the folder `./data/`.

> **Tip:** Use `python data_loading.py -h` to see more options

### Train models


After downloading the data, the `./train.py` file can then be used to train a model.

* Train RESAUG5 trained for 5 more epochs (**RESAUG10**), producing a file represented as `RESAUG10_MODEL_NAME`: 

`python train.py --data_path="data" --num_epochs=5 --model=resnet_augmented --model_state_file="RES5_MODEL_NAME" --resume_training` 
 

* Train demo model (**DEMO**, not used in final report), producing a file represented as `DEMO_MODEL_NAME`: 

`python train.py --data_path="data" --num_epochs=5 --model=demo --model_state_file="model_demo.pt"` 


* Train ResNet-18 trained for 5 epochs (**RES5**), producing a file represented as `RES5_MODEL_NAME`: 

`python train.py --data_path="data" --num_epochs=5 --model=resnet --model_state_file="model_resnet.pt"` 

* Train ResNet-18 with data augmentation for 5 epochs (**RESAUG5**), producing a file represented as `RESAUG5_MODEL_NAME`: 

`python train.py --data_path="data" --num_epochs=5 --model=resnet_augmented --model_state_file="model_resnet_aug.pt"` 


* Train VGG-16 for 5 epochs (**VGG5**), producing a file represented as `VGG5_MODEL_NAME`: 

` python train.py --data_path="data" --num_epochs=5 --model=vgg16_pretrained --model_state_file="model_vgg.pt"` 

> **Tip:** Use `python train.py -h` to see more options

### Test models
After the models are trained and saved, the `./test.py` file can then be used to test a model. 

* Test demo model (not used in final report): 

`python test.py --data_path="data" --model=demo --model_state_file="DEMO_MODEL_NAME"` 

* Test ResNet-18 trained for 5 epochs (**RES5**): 

`python test.py --data_path="data" --model=resnet --model_state_file="RES5_MODEL_NAME"` 

* Test ResNet-18 with data augmentation, trained for 5 epochs (**RESAUG5**): 

`python test.py --data_path="data" --model=resnet_augmented --model_state_file="RESAUG5_MODEL_NAME"` 

* Test RESAUG5 trained for 5 more epochs (**RESAUG10**): 

`python test.py --data_path="data" --model=resnet_augmented --model_state_file="RESAUG10_MODEL_NAME"` 

* Test VGG-16 (**VGG5**): 

`python test.py --data_path="data" --model=vgg16_pretrained --model_state_file="VGG5_MODEL_NAME"` 

> **Tip:** Use `python test.py -h` to see more options

## Saved models
The saved trained models can be found in the following links:
* [RES5](https://drive.google.com/file/d/1u0heklzsMb65usgu9DeNgk90nEphV4K4/view?usp=sharing)
* [RESAUG5](https://drive.google.com/file/d/12h7zSOUw1FqdMRfRYhn8RIKQiysbjyeo/view?usp=sharing)
* [RESAUG10](https://drive.google.com/file/d/1IsBM3rn0q23qyoHgOckXi5QoAi3w-ccn/view?usp=sharing)
* [VGG5](https://drive.google.com/file/d/1-ePhNFry-z_f5DNbxCxBN9BDMFxRb2a6/view?usp=sharing)

## Dataset link
The link to the dataset can be found [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data).

## Used Resources
The resources used in this project are listed below.
### Data source
* [Dataset from Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)
### Code references
* The DEMO network is taken mainly from [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).
* [Transfer Learning using VGG16 in Pytorch | VGG16 Architecture](https://www.analyticsvidhya.com/blog/2021/06/transfer-learning-using-vgg16-in-pytorch/)
* [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
### Python & computational packages
* [Pytorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html)
* [Kaggle API in Python](https://github.com/Kaggle/kaggle-api)
* Numerical and image processing packages: Numpy, Pandas, Matplotlib, PIL
### Computational environments
* Google Colab
