import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
import torch
from sklearn.cluster import KMeans, SpectralClustering

from sentence_transformers import SentenceTransformer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['ID'] = range(1, len(df) + 1)  # Anonymize students with IDs
    cols = list(df.columns)
    cols = cols[-1:] + cols[:-1] # make weaknesses the last column
    return df[cols]

def encode_answers_sbert(model, df):
    student_embeddings = []
    responses = df.drop(['Name','Email'], axis=1)

    for _, row in responses.iterrows():
        response_embeddings = model.encode(row, convert_to_tensor=True)
        
        student_embeddings.append(torch.flatten(response_embeddings))  # Flatten question embeddings
    
    feature_matrix = torch.stack(student_embeddings)
    return feature_matrix.cpu()  # (num_students, embedding_size * num_questions)


def custom_distance(vec1, vec2, weak_weight=0.5):
    # last 384 are weaknesses
    base_sim = cosine(vec1[:-384], vec2[:-384]) # 0 if same, 1 if different
    weak_sim = 1 - cosine(vec1[-384:], vec2[-384:]) # 1 if same, 0 if different

    sim = min(1.0, base_sim + (weak_weight * weak_sim))
    return sim

    
def cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4):
    num_clusters = len(df) // min_group_size  # Ensure enough clusters for min size

    distance_matrix = squareform(pdist(feature_matrix, metric=custom_distance))
    
    # Apply Spectral clustering
    cluster = SpectralClustering(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = cluster.fit_predict(distance_matrix)
    
    
    
    # # Ensure students with similar weaknesses are not grouped together
    # df['Weaknesses'] = df['Weaknesses'].astype(str)
    # unique_weaknesses = df['Weaknesses'].unique()
    
    # for weakness in unique_weaknesses:
    #     students_with_weakness = df[df['Weaknesses'] == weakness]
    #     if len(students_with_weakness) > 1:
    #         clusters_assigned = students_with_weakness['Cluster'].unique()
    #         if len(clusters_assigned) == 1:
    #             # Reassign students with the same weakness to different clusters
    #             for idx, student in enumerate(students_with_weakness.index):
    #                 df.at[student, 'Cluster'] = (df.at[student, 'Cluster'] + idx) % num_clusters
    
    # Ensure groups have a min of 3 and max of 4 students
    clustered_students = df.groupby('Cluster').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    new_clusters = []
    cluster_counter = 0
    temp_group = []
    
    for _, row in clustered_students.iterrows():
        temp_group.append((cluster_counter, row['Name']))
        if len(temp_group) == max_group_size:
            new_clusters.extend(temp_group)
            temp_group = []
            cluster_counter += 1
    
    # If any remaining students, distribute them to ensure min size of 3
    if len(temp_group) >= min_group_size:
        new_clusters.extend(temp_group)
    else:
        for i in range(len(temp_group)):
            new_clusters[i % cluster_counter].append(temp_group[i])
    
    df_clustered = pd.DataFrame(new_clusters, columns=['Cluster', 'Name'])
    return df_clustered

def generate_report(df, output_path):
    df.to_csv(output_path, index=False)
    print(f'Group assignments saved to {output_path}')

# Main Execution
seed=42
torch.manual_seed(seed)
model = SentenceTransformer("all-MiniLM-L6-v2")
file_path = 'synthetic_student_availability_open_ended.csv'
df = preprocess_data(file_path)
feature_matrix = encode_answers_sbert(model, df)
df = cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4)
generate_report(df, 'student_group_assignments_sbert.csv')