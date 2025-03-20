
import random
import torch
from matchmaking import preprocess_data, cluster_students, generate_report, bge_encode, generate_ngrams, create_feature_matrix, sbert_encode, word2vec_encode, cluster_students_2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from itertools import combinations

def evaluate_cosine_similarity(feature_matrix, df):
    similarity_matrix = cosine_similarity(feature_matrix)
    avg_similarity = np.mean([similarity_matrix[i, j] 
                              for i in range(len(df)) 
                              for j in range(i+1, len(df)) if df['Cluster'][i] == df['Cluster'][j]])
    print(f"Average Intra-Cluster Cosine Similarity: {avg_similarity:.4f}")
    return avg_similarity
    
def evaluate_jaccard_similarity(df, original_df, column_name):
    clusters = df['Cluster'].values
    unique_clusters = np.unique(clusters)
    scores = []
    
    for cluster in unique_clusters:
        group = df[df['Cluster'] == cluster]
        student_names = group['Name'].values  # Get student names in this cluster
        
        matched_weaknesses = original_df.set_index('Name').loc[student_names, column_name].astype(str).values  # Match weaknesses using names
        unique_words = set()
        for weakness in matched_weaknesses:
            unique_words.update(weakness.split())  # Dynamically update per method/cluster
        unique_words = list(unique_words)
        
        # Convert each student's weaknesses into a binary vector
        vectorized_values = [[1 if word in v.split() else 0 for word in unique_words] for v in matched_weaknesses]

        if len(vectorized_values) > 1:
            pairwise_scores = [jaccard_score(a, b, average='macro') for a, b in combinations(vectorized_values, 2)]
            if pairwise_scores:
                scores.append(np.mean(pairwise_scores))  # Average Jaccard within cluster
    
    avg_score = np.mean(scores) if scores else 0
    print(f"Jaccard Similarity for {column_name}: {avg_score:.4f}")
    return avg_score

def plot_evaluation_results(models, scores_1, scores_2, file):
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, scores_1, width, label='KMeans Clustering')
    bars2 = ax.bar(x + width/2, scores_2, width, label='Spectral Clustering')
    
    ax.set_xlabel("Embedding Methods")
    ax.set_ylabel("Similarity Scores")
    ax.set_title("Evaluation of Different Embedding Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(models.keys())
    ax.legend()
    
    plt.xticks(rotation=45)
    
    plt.savefig(f'evaluation_plot_{file}.png')  # Save the figure
    plt.close()  # Close to prevent memory leaks


seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# file_path = ['synthetic_student_availability_open_ended.csv', 'Generated-Responses.csv']
file_path = ['Generated-Responses.csv']
for file, path in enumerate(file_path):
    print(f'\n=============FILE {file}: {path}===============\n')
    df = preprocess_data(path)

    encoding_models = {
        "BGE":bge_encode, 
        "N-Grams":create_feature_matrix, 
        "SBERT":sbert_encode, 
        "Word2Vec":word2vec_encode
        }
    cosine_scores = []
    cosine_scores_2 = []
    jaccard_scores = []
    jaccard_scores_2 = []

    input_df = df.drop(['Name', 'Email'], axis=1)

    for name, model in encoding_models.items():
        print(f'{name} Evaluations:')
        feature_matrix = model(input_df)
        clusters_df = cluster_students(df, feature_matrix, min_group_size=3, max_group_size=5)
        cosine_scores.append(evaluate_cosine_similarity(feature_matrix, clusters_df))
        jaccard_scores.append(evaluate_jaccard_similarity(clusters_df, df, 'Weaknesses'))

        # spectral cluster method
        clusters2_df = cluster_students_2(df, feature_matrix, min_group_size=3, max_group_size=5)
        cosine_scores_2.append(evaluate_cosine_similarity(feature_matrix, clusters2_df))
        jaccard_scores_2.append(evaluate_jaccard_similarity(clusters2_df, df, 'Weaknesses'))

        generate_report(clusters_df, f'groups_{file}_{name}_kmeans')
        generate_report(clusters2_df, f'groups_{file}_{name}_spectral')

    print("\nSaving plots...")
    plot_evaluation_results(encoding_models, cosine_scores, cosine_scores_2, f'{file}_cosine')
    plot_evaluation_results(encoding_models, jaccard_scores, jaccard_scores_2, f'{file}_jaccard')
    print("Plot saved")

# generate_report(df, 'student_group_assignments_v3.csv')