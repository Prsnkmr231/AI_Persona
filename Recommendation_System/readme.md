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
```




