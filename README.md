# Eyelid Detection
==============================

## Project Description
This project dedicates to automatically measure key facial measurements based on patients facial images.


## Projcet Orgnization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── labels         <- Data labels, including the mapping relation from old file names to numberred file names, model xml files, etc.
    │   ├── final_outcome  <- Folder to store the final outcome
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │    ├── left
    │   │    └── right
    │   ├── resources 		<- Resources from existing works to aid our code.
    │   │    └── haarcascade    <- Pretrained eye detection model from OpenCV
    │   └── raw            <- The original, immutable data dump.
    │        ├── left
    │        └── right
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── resouces            <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── labeling       <- Scripts to label images
    │   │   ├── labeling_images_left.py
    │   │   └── labeling_images_right.py
    │   │
    │   └── facial_landmark <- Scripts related to facial landmark detection
    │
    │
    └── models                          <- Information related to the model trained by us
        └── landmark_model_parameters   <- Previous trained parameters, from Jacob dataset


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


-------

### Required data before running:
- [68 facial landmark predictor](https://drive.google.com/file/d/1qzEiAi0rE3RLsMhgHepOuga0IiBt0Hg0/view?usp=sharing) ~> data/resources/
- Raw data ~> data/raw/
