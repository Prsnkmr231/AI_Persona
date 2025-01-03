
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os



def dataframe_loader(path):
    df = pd.read_csv(path)
    return df

def positive_labelled_df(user_df,jobs_df,csv_file="positive_labelled_data.csv"):
        results =[]
        jobs_df['job_title'] = jobs_df['job_title'].fillna('').astype(str)
        jobs_df['skills'] = jobs_df['skills'].fillna('').astype(str)
        jobs_df['job_sentence'] = jobs_df[['job_title',  'skills']].agg(' '.join, axis=1)
        jobs_df['job_sent_embeddings'] = jobs_df['job_sentence'].apply(lambda x: model.encode(x))
        job_embeddings = jobs_df['job_sent_embeddings'].tolist()
        print(f"job_embeddings generated")

        user_df['role'] = user_df['role'].fillna('').astype(str)
        user_df['skills'] = user_df['skills'].fillna('').astype(str)
        user_df['bio'] = user_df['bio'].fillna('').astype(str)
        user_df['user_sentence'] = user_df[['role', 'skills', 'bio']].agg(' '.join, axis=1)
        user_df['user_sent_embeddings'] = user_df['user_sentence'].apply(lambda x: model.encode(x))
        print(f"user_embeddings generated")

        user_embeddings = user_df['user_sent_embeddings'].tolist()

        for user_index, user_embedding in enumerate(user_embeddings):
            user_id = user_df.iloc[user_index]['user_sentence']  # or use another identifier if available

            similarities = cosine_similarity(user_embedding.reshape(1, -1), job_embeddings).flatten()

            job_similarity_df = pd.DataFrame({
                'job': jobs_df['job_sentence'],
                'similarity': similarities
            })
        
            top_similar_jobs = job_similarity_df.nlargest(5, 'similarity')

            for _, row in top_similar_jobs.iterrows():
                    results.append({'user': user_id, 'job': row['job'], 'label': 1})

        results_df = pd.DataFrame(results)  
        if os.path.isfile(csv_file): 
            results_df.to_csv(csv_file, mode='a', header=False, index=False) 
        else: 
            results_df.to_csv(csv_file, index=False)


def negative_labelled_df(user_df, jobs_df, all_jobs_dfs, csv_file="negative_labelled_data.csv"):
        results = []
        
        # Check if the jobs_df is in all_jobs_dfs and create a new list with remaining job DataFrames
        remaining_jobs_dfs = [df for df in all_jobs_dfs if not df.equals(jobs_df)]
        
        # Combine the remaining jobs DataFrames into a single DataFrame
        combined_jobs_df = pd.concat(remaining_jobs_dfs, ignore_index=True)

        combined_jobs_df['job_title'] = combined_jobs_df['job_title'].fillna('').astype(str)
        combined_jobs_df['skills'] = combined_jobs_df['skills'].fillna('').astype(str)
        combined_jobs_df['job_sentence'] = combined_jobs_df[['job_title', 'skills']].agg(' '.join, axis=1)
        combined_jobs_df['job_sent_embeddings'] = combined_jobs_df['job_sentence'].apply(lambda x: model.encode(x))
        job_embeddings = combined_jobs_df['job_sent_embeddings'].tolist()
        print(f"Job embeddings generated")

        user_df['role'] = user_df['role'].fillna('').astype(str)
        user_df['skills'] = user_df['skills'].fillna('').astype(str)
        user_df['bio'] = user_df['bio'].fillna('').astype(str)
        user_df['user_sentence'] = user_df[['role', 'skills', 'bio']].agg(' '.join, axis=1)
        user_df['user_sent_embeddings'] = user_df['user_sentence'].apply(lambda x: model.encode(x))
        print(f"User embeddings generated")

        user_embeddings = user_df['user_sent_embeddings'].tolist()

        # Compute similarities and find least similar jobs
        for user_index, user_embedding in enumerate(user_embeddings):
            user_id = user_df.iloc[user_index]['user_sentence']  # or use another identifier if available

            similarities = cosine_similarity(user_embedding.reshape(1, -1), job_embeddings).flatten()

            job_similarity_df = pd.DataFrame({
                'job': combined_jobs_df['job_sentence'],
                'similarity': similarities
            })

            least_similar_jobs = job_similarity_df.nsmallest(5, 'similarity')

            for _, row in least_similar_jobs.iterrows():
                results.append({'user': user_id, 'job': row['job'], 'label': 0})

        results_df = pd.DataFrame(results)

        if os.path.isfile(csv_file):
            results_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_file, index=False)



if os.path.exists("upd_labelled_data.csv"):
    print(f'upda_labelled_data.csv file exists.')

else:
    back_dev_user_df = dataframe_loader('user_seperate_data/Backend Developer_user_Data.csv')
    cloud_user_df = dataframe_loader('user_seperate_data/Cloud Engineer_user_Data.csv')
    content_user_df = dataframe_loader('user_seperate_data/Content Creator_user_Data.csv')
    embedding_user_df = dataframe_loader('user_seperate_data/Embedding Specialist_user_Data.csv')
    graphic_user_df = dataframe_loader('user_seperate_data/Graphic Designer_user_Data.csv')
    java_user_df = dataframe_loader('user_seperate_data/Java Developer_user_Data.csv')
    market_user_df = dataframe_loader('user_seperate_data/Marketing_user_Data.csv')
    python_user_df = dataframe_loader('user_seperate_data/Python Developer_user_Data.csv')
    software_user_df = dataframe_loader('user_seperate_data/Software Developer_user_Data.csv')
    test_user_df = dataframe_loader('user_seperate_data/Test Engineer_user_Data.csv')

    back_dev_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_backend_developer.csv')
    cloud_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_cloud.csv')
    content_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_content_creator.csv')
    embedded_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_embedded.csv')
    graphic_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_Graphoc_designer.csv')
    java_jobs_df    = dataframe_loader('jobs_seperate_data/jobs_data_java.csv')
    market_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_marketing.csv')
    python_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_python.csv')
    software_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_soft_dev.csv')
    test_jobs_df = dataframe_loader('jobs_seperate_data/jobs_data_test_engineer.csv')


    print("Model loading started")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model is loaded....")

    


    positive_labelled_df(back_dev_user_df,back_dev_jobs_df)
    print(f"backend_dev_Completed")
    positive_labelled_df(cloud_user_df,cloud_jobs_df)
    print(f"cloud completed")
    positive_labelled_df(content_user_df,content_jobs_df)  
    print(f"content completed")
    positive_labelled_df(embedding_user_df,embedded_jobs_df)
    print(f"embedding completed")
    positive_labelled_df(graphic_user_df,graphic_jobs_df)
    print(f"graphic completed")
    positive_labelled_df(java_user_df,java_jobs_df)
    print(f"java completed")
    positive_labelled_df(market_user_df,market_jobs_df)
    print(f"market completed")
    positive_labelled_df(python_user_df,python_jobs_df)  
    print(f"python completed")
    positive_labelled_df(software_user_df,software_jobs_df)  
    print("software completed")
    positive_labelled_df(test_user_df,test_jobs_df)  
    print(f"test completed")


    all_jobs_dfs = [
        back_dev_jobs_df, cloud_jobs_df, content_jobs_df, embedded_jobs_df,
        graphic_jobs_df, java_jobs_df, market_jobs_df, python_jobs_df,
        software_jobs_df, test_jobs_df
    ]



    


    negative_labelled_df(back_dev_user_df, back_dev_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Backend Developer user")

    negative_labelled_df(cloud_user_df, cloud_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Cloud Engineer user")

    (content_user_df, content_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Content Creator user")

    negative_labelled_df(embedding_user_df, embedded_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Embedding Specialist user")

    negative_labelled_df(graphic_user_df, graphic_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Graphic Designer user")

    negative_labelled_df(java_user_df, java_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Java Developer user")

    negative_labelled_df(market_user_df, market_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Marketing user")

    negative_labelled_df(python_user_df, python_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Python user")

    negative_labelled_df(software_user_df, software_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Software Developer user")

    negative_labelled_df(test_user_df, test_jobs_df, all_jobs_dfs)
    print(f"Negative labels generated for Test Engineer user")

    pos_df = pd.read_csv("positive_labelled_data.csv")
    neg_df = pd.read_csv("negative_labelled_data.csv")

    combined_df = pd.concat([pos_df, neg_df], ignore_index=True) 
    # Shuffle the combined dataframe 
    labelled_data = combined_df.sample(frac=1).reset_index(drop=True)


    labelled_data.to_csv("upd_labelled_data.csv")
    print(f'upd_labelled_data.csv file was generated successfully')