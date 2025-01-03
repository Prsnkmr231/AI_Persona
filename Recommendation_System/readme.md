# Recommendation System

## Overview

This repository contains two projects:

1. **DistilBert Recommendation System**: A system that uses DistilBert for job recommendations and user profile matching.
2. **SentenceTransformer Recommendation API**: A REST API built with FastAPI that uses the SentenceTransformer model for course and job recommendations based on user input using pre-trained embeddings and cosine similarity.

Both projects are integrated within a unified folder structure.

## Features

- **DistilBert Recommendation System**:
  - Job recommendation system using DistilBert for various job roles and user profiles.
  - Includes training, testing, and data preprocessing scripts for the DistilBert model.
  
- **SentenceTransformer Recommendation API**:
  - Fast and lightweight REST API for recommending courses and jobs.
  - Uses the SentenceTransformer model for generating embeddings and cosine similarity-based recommendations.
  - Provides recommendations based on user input (skills, experience).
  - Supports both query and JSON-based input.
  - Configurable via a `config.json` file.

## Folder Structure

```plaintext
recommendation_system_and_api/
├── distilbert_recommendation/
│   ├── jobs_seperate_data/
│   ├── user_seperate_data/
│   ├── DistilBert_Testing.py
│   ├── Distilbert_Training.ipynb
│   ├── master_jobs_files.csv
│   ├── training_data_generator.py
│   ├── positive_labelled_data.csv
│   ├── negative_labelled_data.csv
├── sentence_transformer_api/
│   ├── data/
│   │   ├── course_embeddings.npy
│   │   ├── Jobs_embeddings.npy
│   │   ├── master_jobs_file.csv
│   │   ├── online_courses.csv
│   ├── src/
│   │   ├── courses_recommender.py
│   │   ├── jobs_recommender.py
│   │   ├── utils.py
│   ├── config.json
│   ├── main.py
│   ├── api.py
│   └── README.md




Folder and File Descriptions
distilbert_recommendation/
This folder contains files related to training and testing a recommendation system based on the DistilBert model.

jobs_seperate_data/: Contains CSV files with job data for specific roles such as backend developer, content creator, etc.
user_seperate_data/: Contains CSV files with user profiles for specific roles such as backend developer, content creator, etc.
DistilBert_Testing.py: Python script for testing the DistilBert model checkpoints.
Distilbert_Training.ipynb: Jupyter notebook for training the DistilBert model.
master_jobs_files.csv: CSV file with all jobs related to the roles from the jobs_seperate_data folder.
training_data_generator.py: Python script for generating training data (positive and negative labelled data).
positive_labelled_data.csv: CSV file with positive labelled job data (jobs belonging to the user profiles).
negative_labelled_data.csv: CSV file with negative labelled job data (jobs that do not belong to the user profiles).
sentence_transformer_api/
This folder contains files related to the SentenceTransformer-based recommendation system and FastAPI server.

data/: Contains pre-trained embeddings and CSV files for courses and jobs.
course_embeddings.npy: Numpy array containing embeddings for courses.
Jobs_embeddings.npy: Numpy array containing embeddings for jobs.
master_jobs_file.csv: CSV file with a master list of job listings.
online_courses.csv: CSV file with details of online courses.
src/: Core scripts for recommendation logic.
courses_recommender.py: Script for recommending courses based on user input.
jobs_recommender.py: Script for recommending jobs based on user input.
utils.py: Utility functions used by the recommender scripts.
config.json: Configuration file for setting up the API and paths.
main.py: Entry point for running the FastAPI server.
api.py: Defines the API endpoints for the recommendation system.
Setup
Prerequisites
Python 3.x
Required Python packages (listed in requirements.txt for both projects)
Installation
Clone the Repository:

bash
Copy code
git clone <repository-url>
cd recommendation_system_and_api
Install Dependencies: Navigate to both distilbert_recommendation/ and sentence_transformer_api/ and install the required packages listed in their respective requirements.txt.
