
from matchmaking import preprocess_data, cluster_students, generate_report, bge_encode, generate_ngrams, create_feature_matrix, sbert_encode, word2vec_encode
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
                              for j in range(len(df)) if df['Cluster'][i] == df['Cluster'][j]])
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

def plot_evaluation_results(models, cosine_scores, jaccard_scores):
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, cosine_scores, width, label='Cosine Similarity')
    bars2 = ax.bar(x + width/2, jaccard_scores, width, label='Jaccard Similarity')
    
    ax.set_xlabel("Embedding Methods")
    ax.set_ylabel("Similarity Scores")
    ax.set_title("Evaluation of Different Embedding Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.xticks(rotation=45)
    
    plt.savefig("evaluation_plot.png")  # Save the figure
    plt.close()  # Close to prevent memory leaks



file_path = 'synthetic_student_availability_open_ended.csv'
df = preprocess_data(file_path)

models = ["BGE", "N-Grams", "SBERT", "Word2Vec"]
cosine_scores = []
jaccard_scores = []

print("BGE Evaluations:")
bge_feature_matrix = bge_encode(df)
bge_df = cluster_students(df, bge_feature_matrix, min_group_size=3, max_group_size=4)
cosine_scores.append(evaluate_cosine_similarity(bge_feature_matrix, bge_df))
jaccard_scores.append(evaluate_jaccard_similarity(bge_df, df, 'Weaknesses'))

print("\nNGram Evaluations:")
ngram_feature_matrix, text_data = create_feature_matrix(df, n=3)
ngram_df = cluster_students(df, ngram_feature_matrix, min_group_size=3, max_group_size=4)
cosine_scores.append(evaluate_cosine_similarity(ngram_feature_matrix, ngram_df))
jaccard_scores.append(evaluate_jaccard_similarity(ngram_df, df, 'Weaknesses'))

print("\nSBERT Evaluations:")
model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_feature_matrix = sbert_encode(model, df)
sbert_df = cluster_students(df, sbert_feature_matrix, min_group_size=3, max_group_size=4)
cosine_scores.append(evaluate_cosine_similarity(sbert_feature_matrix, sbert_df))
jaccard_scores.append(evaluate_jaccard_similarity(sbert_df, df, 'Weaknesses'))

print("\nWord2Vec Evaluations:")
word2vec_feature_matrix = word2vec_encode(df)
word2vec_df = cluster_students(df, word2vec_feature_matrix, min_group_size=3, max_group_size=4)
cosine_scores.append(evaluate_cosine_similarity(word2vec_feature_matrix, word2vec_df))
jaccard_scores.append(evaluate_jaccard_similarity(word2vec_df, df, 'Weaknesses'))

print("\nSaving plot...")
plot_evaluation_results(models, cosine_scores, jaccard_scores)
print("Plot saved")

# generate_report(df, 'student_group_assignments_v3.csv')