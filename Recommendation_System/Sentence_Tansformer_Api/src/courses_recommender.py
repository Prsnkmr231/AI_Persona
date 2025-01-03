import os
from .utils import load_dataframe, clean_text, load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Course_Recommender:
    def __init__(self, config):
        self.config = config
        courses_file_path = os.path.join(self.config["data_folder_path"], "online_courses.csv")
        self.master_courses_df = load_dataframe(courses_file_path)

    def preprocess_df(self):
        self.master_courses_df[['Title', 'Skills', 'Short Intro']] = self.master_courses_df[['Title', 'Skills', 'Short Intro']].fillna('')
        self.master_courses_df["course_sentence"] = (
            self.master_courses_df['Title'] + ' ' +
            self.master_courses_df['Skills'] + ' ' +
            self.master_courses_df['Short Intro']
        ).apply(clean_text)

    def generate_embeddings(self, data_path):
        course_embeddings_path = os.path.join(data_path, "course_embeddings.npy")
        if os.path.exists(course_embeddings_path):
            self.model = load_model(self.config['model_name'])
            self.embedding_matrix = np.load(course_embeddings_path)
        else:
            self.model = load_model(self.config['model_name'])
            self.master_courses_df['embeddings'] = self.master_courses_df['course_sentence'].apply(lambda x: self.model.encode(x))
            self.embedding_matrix = np.vstack(self.master_courses_df['embeddings'].values)
            np.save(course_embeddings_path, self.embedding_matrix)

    def get_recommendations(self, user_profile):
        user_embedding = self.model.encode(user_profile)
        similarities = cosine_similarity([user_embedding], self.embedding_matrix).flatten()
        similar_indices = similarities.argsort()[-10:][::-1]
        recommended_df = self.master_courses_df[['Title', 'Skills', 'Rating', 'Level']].iloc[similar_indices]
        return recommended_df.to_json(orient='records')

    def recommend_courses(self, user_profile):
        self.preprocess_df()
        self.generate_embeddings(self.config["data_folder_path"])
        return self.get_recommendations(user_profile)
