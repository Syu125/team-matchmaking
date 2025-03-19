import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['ID'] = range(1, len(df) + 1)  # Anonymize students with IDs
    return df

def encode_answers(df):
    text_columns = [col for col in ['Work Preference', 'Assignment Start Time', 'Time Commitment', 'Team Role Preference',
                                    'Team Experience', 'Communication Method', 'Class Goals', 'Strengths', 'Weaknesses'] if col in df.columns]
    
    if not text_columns:
        raise KeyError("None of the expected text columns are present in the dataset.")
    
    # Combine text columns into a single string per student
    # Can skip this and just do it on specific sections 
    df['Combined_Text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    
    stop_words = set(stopwords.words('english'))
    df['Tokenized_Text'] = df['Combined_Text'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words])
    
    model = Word2Vec(df['Tokenized_Text'], vector_size=100, window=5, min_count=1, workers=4)
    
    def vectorize_text(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Use zero vector if no words found
    
    feature_matrix = np.vstack(df['Tokenized_Text'].apply(vectorize_text))
    return feature_matrix

def cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4):
    num_clusters = len(df) // min_group_size  # Ensure enough clusters for min size
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(feature_matrix)
    
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
file_path = 'synthetic_student_availability_open_ended.csv'
df = preprocess_data(file_path)
feature_matrix = encode_answers(df)
df = cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4)
generate_report(df, 'student_group_assignments_word2vec.csv')
