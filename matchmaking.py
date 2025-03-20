import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import nltk
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['ID'] = range(1, len(df) + 1)  # Anonymize students with IDs
    return df

def bge_encode(df):
    text_columns = [col for col in ['Work Preference', 'Assignment Start Time', 'Time Commitment', 'Team Role Preference',
                                    'Team Experience', 'Communication Method', 'Class Goals', 'Strengths', 'Weaknesses'] if col in df.columns]
    
    if not text_columns:
        raise KeyError("None of the expected text columns are present in the dataset.")
    
    # Hugging Face's BGE model
    model = SentenceTransformer("BAAI/bge-base-en")  
    encoded_texts = model.encode(df[text_columns].astype(str).agg(' '.join, axis=1).tolist(), 
                                 normalize_embeddings=True)

    return np.array(encoded_texts)

def generate_ngrams(text, n=3):
    # Handle potential NaN values and convert to string
    if pd.isna(text):
        return ""
    # Convert to string and split into words
    words = str(text).split()
    # If there are fewer words than n, return the text as is
    if len(words) < n:
        return " ".join(words)
    # Generate n-grams manually
    ngrams_list = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams_list.append(ngram)
    return " ".join(ngrams_list)

def create_feature_matrix(df, n=3):
    # Combine all text columns into a single string per student for vectorization
    text_data = []
    for idx, row in df.iterrows():
        combined_text = ""
        for column in df.columns:
            if column != 'ID':
                # Generate n-grams for each cell and combine
                ngrams_text = generate_ngrams(row[column], n)
                combined_text += ngrams_text + " "
        text_data.append(combined_text.strip())
    
    # Use TF-IDF Vectorizer to create a numerical feature matrix
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(text_data).toarray()
    
    return feature_matrix, text_data

def sbert_encode(model, df):
    student_embeddings = []
    for _, row in df.iterrows():
        response_embeddings = (model.encode(row, convert_to_tensor=True))
        student_embeddings.append(torch.flatten(response_embeddings))  # Flatten question embeddings
    
    return torch.stack(student_embeddings).cpu()  # (num_students, embedding_size * num_questions)

def word2vec_encode(df):
    text_columns = [col for col in ['Work Preference', 'Assignment Start Time', 'Time Commitment', 'Team Role Preference',
                                    'Team Experience', 'Communication Method', 'Class Goals', 'Strengths', 'Weaknesses'] if col in df.columns]
    
    if not text_columns:
        raise KeyError("None of the expected text columns are present in the dataset.")
    
    # Combine text columns into a single string per student
    df['Combined_Text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    
    stop_words = set(stopwords.words('english'))
    df['Tokenized_Text'] = df['Combined_Text'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words])
    
    # Train Word2Vec model on the tokenized text
    model = Word2Vec(sentences=df['Tokenized_Text'].tolist(), vector_size=100, window=5, min_count=1, workers=4, sg=1)
    
    def vectorize_text(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)  # Use zero vector if no words found
    
    feature_matrix = np.vstack(df['Tokenized_Text'].apply(vectorize_text))
    feature_matrix = normalize(feature_matrix, axis=1)  # Normalize to ensure proper cosine similarity
    
    return feature_matrix

def cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4):
    num_clusters = len(df) // min_group_size  # Ensure enough clusters for min size
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(feature_matrix)
    
    # Ensure students with similar weaknesses are not grouped together
    df['Weaknesses'] = df['Weaknesses'].astype(str)
    unique_weaknesses = df['Weaknesses'].unique()
    
    for weakness in unique_weaknesses:
        students_with_weakness = df[df['Weaknesses'] == weakness]
        if len(students_with_weakness) > 1:
            clusters_assigned = students_with_weakness['Cluster'].unique()
            if len(clusters_assigned) == 1:
                # Reassign students with the same weakness to different clusters
                for idx, student in enumerate(students_with_weakness.index):
                    df.at[student, 'Cluster'] = (df.at[student, 'Cluster'] + idx) % num_clusters
    
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
# file_path = 'synthetic_student_availability_open_ended.csv'
# df = preprocess_data(file_path)
# feature_matrix = encode_answers(df)
# df = cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4)
# generate_report(df, 'student_group_assignments_v3.csv')