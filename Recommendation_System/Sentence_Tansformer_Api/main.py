
import argparse
import os
from src.jobs_recommender import Jobs_recommender
from src.courses_recommender import Course_Recommender
import json


def load_config(config_file_path):  
    with open(config_file_path, 'r') as config_file: 
        config = json.load(config_file) 
    return config

def main(config):
    if config["jobs"]==True:
        user_profile = 'Python Backend Developer,"Python, Django, Flask, PostgreSQL, REST APIs, Git",Python backend developer experienced in building and maintaining scalable web applications using Django and Flask. She is skilled in database management with PostgreSQL and has worked extensively with RESTful API design.'
        jobs_recommender = Jobs_recommender(config)
        jobs_data = jobs_recommender.recommend_jobs(user_profile)
        print(jobs_data)

    if config["courses"]==True:
        user_profile = "Python, Django, Flask, PostgreSQL, REST APIs, Git"
        courses_recommender = Course_Recommender(config)
        courses_data = courses_recommender.recommend_courses(user_profile)
        print(courses_data)


if __name__ == "__main__":
    # print(os.getcwd())
    config_path = os.path.join(os.getcwd(),"config.json")
    config = load_config(config_path)
    main(config)