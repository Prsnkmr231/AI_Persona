Recommendation system is a folder containing the files and folders mentioned below
# DistilBert

## Overview

The DistilBert project is designed to create a chatbot that uses DistilBert for various applications such as job recommendation and QA. The project contains training and testing scripts, data preprocessing scripts, and datasets related to different job roles and user profiles.

## Folder Structure

    DistilBert/
    ├── jobs_seperate_data/
    ├── user_seperate_data/
    ├── DistilBert_Testing.py
    ├── Distilbert_Training.ipynb
    ├── master_jobs_files.csv
    ├── training_data_generator.py
    ├── positive_labelled_data.csv
    ├── negetive_labelled_data.csv

### Folder and File Descriptions

- **jobs_seperate_data/**: This folder contains different CSV files, each representing jobs related to a specific role such as backend developer, content creator, etc.
- **user_seperate_data/**: This folder contains different CSV files, each representing user profiles related to a specific role such as backend developer, content creator, etc.
- **DistilBert_Testing.py**: A Python script for testing the checkpoints generated after training the DistilBert model.
- **Distilbert_Training.ipynb**: A Jupyter notebook for training the DistilBert model.
- **master_jobs_files.csv**: A single CSV file that contains all the jobs related to the roles present in the `jobs_seperate_data` folder.
- **training_data_generator.py**: A Python script that uses the CSV files in `jobs_seperate_data` and `user_seperate_data` to generate `positive_labelled_data.csv` and `negative_labelled_data.csv`. It then concatenates and shuffles these values to create `upd_labelled_data.csv`.
- **positive_labelled_data.csv**: A CSV file containing positive labelled data (label 1) of jobs that belong to the user profiles.
- **negative_labelled_data.csv**: A CSV file containing negative labelled data (label 0) of jobs that do not belong to the user profiles.

## Setup and Usage

### Prerequisites
- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd DistilBert
