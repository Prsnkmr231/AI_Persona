# Recommendation API

## Overview
This project implements a recommendation system using FastAPI to provide course and job recommendations based on user input. The system leverages pre-trained embeddings and cosine similarity for recommendations.

## Features
- Recommends courses or jobs based on user-provided input.
- Lightweight and fast REST API built with FastAPI.
- Supports both query and JSON-based input.
- Easily configurable with a `config.json` file.

## Folder Structure

    ├── data/
    │   ├── course_embeddings.npy
    │   ├── Jobs_embeddings.npy
    │   ├── master_jobs_file.csv
    │   ├── online_courses.csv
    ├── src/
    │   ├── courses_recommender.py
    │   ├── jobs_recommender.py
    │   ├── utils.py
    ├── config.json
    ├── main.py
    ├── api.py
    └── README.md

### Folder and File Descriptions

- **data/**: Contains pre-trained embeddings and CSV files for courses and jobs.
  - **course_embeddings.npy**: Numpy array of embeddings for courses.
  - **Jobs_embeddings.npy**: Numpy array of embeddings for jobs.
  - **master_jobs_file.csv**: CSV file with master job listings.
  - **online_courses.csv**: CSV file with online courses information.
  
- **src/**: Contains the core scripts for recommendation logic.
  - **courses_recommender.py**: Script to recommend courses based on user input.
  - **jobs_recommender.py**: Script to recommend jobs based on user input.
  - **utils.py**: Utility functions used by the recommender scripts.
  
- **config.json**: Configuration file for setting up the API and paths.
- **main.py**: Entry point for running the FastAPI server.
- **api.py**: Defines the API endpoints for the recommendation system.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Recommendation_API

2.**Usage** 

**Request Body**

{
  "user_skills": "Python, Django, Flask, PostgreSQL, REST APIs, Git",
  "user_id": 2,
  "user_experience": 2.5
}

**Response**
{
    "recommendations": [
        {
            "Title": "Complete Python Developer",
            "Skills": "Python, Django, Flask",
            "Rating": 4.7,
            "Level": "Intermediate"
        },
        ...
    ]
}


