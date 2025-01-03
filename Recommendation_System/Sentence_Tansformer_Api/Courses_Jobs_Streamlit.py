import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st



# """

# Documentation:

#     Input:
#     - job_title: Text input for the user's job title.
#     - years_exp: Text input for the user's years of experience.
#     - skills: Text input for the user's skills.

#     Output:
#     - Recommended Courses: A DataFrame displayed in Streamlit with course recommendations based on the user's profile.
#     - Recommended Jobs: A DataFrame displayed in Streamlit with job recommendations based on the user's profile.

#     Functionality:
#     1. Load Model and Data:
#     - The load_model function loads a pre-trained SentenceTransformer model and caches it.
#     - The load_data function loads pre-processed course data, embeddings, original courses, master job data, and job embeddings, and caches them.

#     2. Recommendation Functions:
#     - The get_recommended_courses function generates course recommendations based on the user's profile by computing cosine similarities between user profile embeddings and course embeddings.
#     - The get_recommended_jobs function generates job recommendations based on the user's profile by computing cosine similarities between user profile embeddings and job embeddings.
#     - The extract_bounds function extracts the lower bound of experience from a string.

#     3. Streamlit Interface:
#     - The Streamlit interface allows users to input their job title, years of experience, and skills.
#     - When the "Get Recommendations" button is clicked, the script generates and displays recommended courses and jobs based on the user's profile.
# """



# Set Streamlit page configuration
st.set_page_config(layout="wide")



@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    course_embeddings = np.load('data/course_embeddings.npy')
    orig_courses_df = pd.read_csv('data/Online_Courses.csv')
    master_jobs_df = pd.read_csv("data/master_jobs_file.csv")
    jobs_embeddings = np.load('data/job_embeddings.npy')
    return course_embeddings, orig_courses_df, course_embeddings, master_jobs_df, jobs_embeddings

# Initialize model and load data
model = load_model()
course_embeddings, orig_courses_df, course_embeddings, master_jobs_df, jobs_embeddings = load_data()

# Function to get recommended courses based on the user profile
def get_recommended_courses(user_profile):
    user_embedding = model.encode(user_profile)
    similarities = cosine_similarity([user_embedding], course_embeddings).flatten()
    top_indices = similarities.argsort()[-15:][::-1]
    recommendations = orig_courses_df[['Title', 'Skills', 'Rating', 'Level']].iloc[top_indices]
    recommendations = recommendations.drop_duplicates(subset=["Title"])
    return recommendations

# Function to get recommended jobs based on the user profile
def get_recommended_jobs(user_profile):
    user_embedding = model.encode(user_profile)
    similarities = cosine_similarity([user_embedding], jobs_embeddings).flatten()
    top_indices = similarities.argsort()[-20:][::-1]
    return master_jobs_df[['job_title', 'company_name', 'experience', 'location', 'job_description', 'skills']].iloc[top_indices]

# Function to extract lower bound of experience from string
def extract_bounds(experience_str):
    if 'Yrs' in experience_str:
        if '-' in experience_str:
            return int(experience_str.split('-')[0])
        return int(experience_str.split()[0])
    if 'month' in experience_str:
        return 0
    return 3

# Streamlit Sidebar for user input
st.title("Recommendations based on user profile")

with st.sidebar:
    st.header("Enter the User Profile details")
    job_title = st.text_input("Enter the job title")
    years_exp = st.text_input('Enter years of experience')
    skills = st.text_input('Your Skills (e.g., machine learning, python, data analysis)', '')

user_profile = job_title + years_exp + skills

# Button to generate recommendations
if st.sidebar.button("Get Recommendations"):
    if user_profile:
        job_recommendations = get_recommended_jobs(user_profile)
        job_recommendations['LowerBound'] = job_recommendations['experience'].apply(extract_bounds)
        filtered_df = job_recommendations[job_recommendations['LowerBound'] <= int(years_exp)]
        
        recommended_courses = get_recommended_courses(skills)

        st.header("Recommended Courses based on the user profile")
        st.dataframe(recommended_courses)

        st.header("Recommended Jobs based on the user profile")
        st.dataframe(filtered_df)
else:
    st.write("Please enter your profile on the left and click 'Get Recommendations'")
