## Project Status: Finished

## Project Intro/Objective

This project, conducted at the University of Applied Sciences and Arts Northwestern Switzerland FHNW, School of Engineering, in Brugg-Windisch, represents a pioneering effort in utilizing machine learning (ML) and deep learning (DL) techniques for the early detection of prolactinomas. Spearheaded by Jan Zwicky, Thomas Mandelz, and guided by Professors Suter and Perruchoud, the study collaborates closely with medical professionals from Kantonsspital Aarau, Dr. Tristan Struja and Dr. Felice Burn.

## Folderoverview

| Folder                          | Subfolders                                  | Description                                                  |
|---------------------------------|---------------------------------------------|--------------------------------------------------------------|
| data                            | test, train                                 | Data for testing and training                                |
| eda                             |                                             | Exploratory Data Analysis (EDA) files, Data processing and partitioning files                  |
| MedicalNet                      | models, utils                               | MedicalNet (Med3d) library components: models and utility functions  |
| modelling                       | combined, mri_data, tabular_data            | Modelling-related folders: combined models, MRI data, tabular data  |
|                                 |   - mri_data: results_augmented_weighted   | MRI data and augmented weighted results                      |
|                                 |   - tabular_data:                          |  tabular data machine learning pipelines                                                            |
|                                 |       - images:                            | Images for tabular data analysis: fn, fp, tn, tp            |
|                                 |       - results_csv                        | CSV results for tabular data analysis                       |
| models                          | saved_models                                | Saved machine learning models                               |
| NODE                            |                                             | NODE deep learning components       |
| raw_data                        | nii_files                                   | raw data for tab and NII files                               |
| src                             |                                             | Source code for deep learning pipeline and related files                                |



### Methods Used

* Deep Learning
* Machine Learning

### Technologies

* Python
* PyTorch
* wandb
* numpy
* pandas
* Azure Machine Learning

## Getting Started

* Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
* Demo files are being kept [here](demo)
* Raw Data is being kept [here](competition_data)
* Explorative Dataanalysis Scripts and Files are being kept [here](Eda)
* Megadetector Scripts and data is being kept [here](megadetector)
* Models are being kept [here](model_submit)
* Models submissions are being kept [here](data_submit)
* Source files for training are being kept [here](modelling)
* Source files for pipeline are being kept [here](src)

## Pipenv for Virtual Environment

### First install of Environment

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv install`
* Restart VS Code
* Choose the newly created "tierli_ahluege" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv sync`

## Contributing Members

* **[Thomas Mandelz](https://github.com/tmandelz)**
* **[Jan Zwicky](https://github.com/swiggy123)**
