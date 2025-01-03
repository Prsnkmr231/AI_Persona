import os
from .utils import load_dataframe, clean_text, load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Jobs_Recommender:
    def __init__(self, config):
        self.config = config
        jobs_file_path = os.path.join(self.config["data_folder_path"], "master_jobs_file.csv")
        self.master_jobs_df = load_dataframe(jobs_file_path)

    def preprocess_df(self):
        self.master_jobs_df[['job_title', 'experience', 'job_description', 'skills']] = self.master_jobs_df[['job_title', 'experience', 'job_description', 'skills']].fillna('')
        self.master_jobs_df["job_sentence"] = (
            self.master_jobs_df['job_title'] + ' ' +
            self.master_jobs_df['experience'] + ' ' +
            self.master_jobs_df['job_description'] + ' ' +
            self.master_jobs_df['skills']
        ).apply(clean_text)

    def generate_embeddings(self, data_path):
        job_embeddings_path = os.path.join(data_path, "job_embeddings.npy")
        if os.path.exists(job_embeddings_path):
            self.model = load_model(self.config['model_name'])
            self.embedding_matrix = np.load(job_embeddings_path)
        else:
            self.model = load_model(self.config['model_name'])
            self.master_jobs_df['embeddings'] = self.master_jobs_df['job_sentence'].apply(lambda x: self.model.encode(x))
            self.embedding_matrix = np.vstack(self.master_jobs_df['embeddings'].values)
            np.save(job_embeddings_path, self.embedding_matrix)

    def get_recommendations(self, user_profile):
        user_embedding = self.model.encode(user_profile)
        similarities = cosine_similarity([user_embedding], self.embedding_matrix).flatten()
        similar_indices = similarities.argsort()[-10:][::-1]
        recommended_df = self.master_jobs_df[['job_title', 'company_name', 'experience', 'skills']].iloc[similar_indices]
        return recommended_df.to_json(orient='records')

    def recommend_jobs(self, user_profile):
        self.preprocess_df()
        self.generate_embeddings(self.config["data_folder_path"])
        return self.get_recommendations(user_profile)
