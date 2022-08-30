# 2022-DL-Final-Project
Also obviously this is a private repo, the submitted files will just the final version of this repo.

## Links
* The Colab demo file can be found [here](https://colab.research.google.com/drive/1S5pJnkNnFQOg1wkDW5q-kJhHEHNOMR9C?usp=sharing).
* The link to the Github repository can be found **THIS NEEDS TO BE ADDED**.

### Python Setup
You can setup your desired python env using conda, virtualenv etc. Make sure to get all the dependencies from `requirements.txt`:

`pip install -r requirements.txt`

### Setup Kaggle for data loading
In order to download the dataset, you need access to Kaggle. Please follow these steps:
1. Create account on [Kaggle](https://www.kaggle.com/)
2. Go to `https://www.kaggle.com/<ACCOUNT_NAME>/account`
2. Click "Create New API Token", which downloads a token file
4. Move downloaded token file to ~/.kaggle/kaggle.json`

If you now call the function `load_data` in `data_loading.py`, the dataset will be downloaded into the `data/` folder in this repo. When training or testing for the first time, you can download the data.

### Setup Wandb
For the training phase, we recommend logging to wandb. For this, please create an account as described [here](https://docs.wandb.ai/quickstart). When training a model, please specify tags in the CLI args in order to help us organize the results.

## Using this tool
### Train a model
Use the `train.py`. Run `python train.py -h` to see the required arguments.

### Colab Demo
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


### Train models
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