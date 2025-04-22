
from fastapi import FastAPI
from pydantic import BaseModel
from src.courses_recommender import Course_Recommender
from src.jobs_recommender import Jobs_Recommender
import os
import json

app = FastAPI()

print(f"testing")

print(f"second change for testing the git")

def load_config(config_file_path="config.json"):
    with open(config_file_path, 'r') as config_file:
        return json.load(config_file)

config = load_config()
course_recommender = Course_Recommender(config)
job_recommender = Jobs_Recommender(config)

class RecommendationRequest(BaseModel):
    user_skills: str
    user_id: int
    user_experience: float
    


@app.post("/recommend_courses")
def recommend(request: RecommendationRequest):
    return {"recommendations": json.loads(course_recommender.recommend_courses(request.user_skills))}

    


@app.post("/recommend_jobs")
def recommend(request: RecommendationRequest):  
    return {"recommendations": json.loads(job_recommender.recommend_jobs(request.user_skills))}
    
