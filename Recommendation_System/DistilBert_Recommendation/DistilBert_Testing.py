import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


# Function to recommend jobs using the fine-tuned model

def recommend_jobs(user_profile, job_listings, top_n=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fine_tuned_model.to(device)
    
        # Encode the user profile
        user_profile_encoded = tokenizer("[USER] " + user_profile, return_tensors="pt", padding=True, truncation=True).to(device)
        scores = []
        counter = 1
        for idx, job in enumerate(job_listings):
            if counter%100==0:
                print(f"job embedding at-{counter}")
            job_encoded = tokenizer("[JOB] " + job, return_tensors="pt", padding=True, truncation=True).to(device)

            inputs = {
                "input_ids": torch.cat((user_profile_encoded['input_ids'], job_encoded['input_ids']), dim=1),
                "attention_mask": torch.cat((user_profile_encoded['attention_mask'], job_encoded['attention_mask']), dim=1),
            }
            with torch.no_grad():
                output = fine_tuned_model(**inputs)
                score = output.logits[0][1].item()  
                scores.append((idx, job, score)) 
            counter+=1

        print(f"embeddings and most similar jobs are generated")

        recommended_jobs = sorted(scores, key=lambda x: x[2], reverse=True)[:top_n]
        print(recommended_jobs)
        return [(idx, job) for idx, job, score in recommended_jobs] 


if os.path.exists('checkpoint_2532'):

    print(f"Model Loading started")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") 
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained('checkpoint_2532') 

    print(f"Model Loading completed")

    if os.path.exists("master_jobs_file.csv"):


        job_listings_df = pd.read_csv("master_jobs_file.csv") 
        
        # Combine job details (role, skills, description) for each job
        job_listings = [
            f"{row['job_title']}, {row['skills']}"
            for _, row in job_listings_df.iterrows()
        ]
    
        user_profile = "Java Developer, with skills javascript,node.js,spring boot,j2ee,core java"

        recommended_jobs = recommend_jobs(user_profile, job_listings, top_n=5)

        result_df = job_listings_df.loc[[idx for idx, _ in recommended_jobs]].copy()
        result_df["Combined Details"] = [job for _, job in recommended_jobs]

        print(result_df.head())

    else:
         print(f"master_jobs_file.csv does not exists,add the master_jobs_file to the directory")

else:
     print(f"checkpoint 2532 is not available in the directory,run the DistilBert_Training.ipynb it will generates the checkpoints")