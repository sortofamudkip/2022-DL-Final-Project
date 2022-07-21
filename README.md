# 2022-DL-Final-Project
 
Code for the DL final project goes here.
Ideally it should mimic the file structure, as described in [page 6 of these slides](https://docs.google.com/presentation/d/1Lbggpj_nj4RomOm4q35XUcoOoDsIDvT18GLpOIygC2Q/edit#slide=id.g8646803fdf_0_15).

Also obviously this is a private repo, the submitted files will just the final version of this repo.

## Links
* The Colab demo file (unfortunately managed by Jake) can be found [here](https://colab.research.google.com/drive/1S5pJnkNnFQOg1wkDW5q-kJhHEHNOMR9C?usp=sharing).
* The docs file can be found [here](https://docs.google.com/document/d/1BD8EmhoiHegGZ0VttSsGSjRnBaHkJEk3UOHgU1ACWv0/edit#).
* Notion site can be found [here](https://brook-emery-b8d.notion.site/DL-Project-Histopathologic-Cancer-Detection-5bf981a7614141de9c55be811333a1d2).

## Required setup
### Setup Anaconda env

### Setup Kaggle for data loading
In order to download the dataset, you need access to Kaggle. Please follow these steps:
1. Create account on [Kaggle](https://www.kaggle.com/)
2. Go to `https://www.kaggle.com/<ACCOUNT_NAME>/account`
2. Click "Create New API Token", which downloads a token file
4. Move downloaded token file to ~/.kaggle/kaggle.json`

If you now call the function `load_data` in `data_loading.py`, we download the dataset the `data` path in this repo. You can run this function as often as you want without redownloading the data.
