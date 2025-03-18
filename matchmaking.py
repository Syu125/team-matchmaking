import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['ID'] = range(1, len(df) + 1)  # Anonymize students with IDs
    return df

#TODO: add ngram to encode the data that is read in

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

def encode_with_ngrams(df, n=3):
    ngram_df = pd.DataFrame()
    ngram_df['ID'] = df['ID']
    for column in df.columns:
        if column != 'ID':
            ngram_df[f'{column}_ngrams'] = df[column].apply(lambda x: generate_ngrams(x, n))
    return ngram_df


def encode_answers(df, n = 3):

    #change to encode by column
    text_columns = [col for col in ['Work Preference', 'Assignment Start Time', 'Time Commitment', 
                                    'Team Role Preference', 'Team Experience', 'Communication Method', 
                                    'Class Goals', 'Strengths', 'Weaknesses'] if col in df.columns]
    
    if not text_columns:
        raise KeyError("None of the expected text columns are present in the dataset.")

    encoded_matrices = []
    
    for col in text_columns:
        word_to_index = {}
        encoded_texts = []
        
        for text in df[col].astype(str):
            encoded_vector = []
            words = text.split()
            for word in words:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index) + 1  # Assign unique index
                encoded_vector.append(word_to_index[word])
            encoded_texts.append(encoded_vector)
        
        # Pad/truncate vectors to a fixed length
        max_length = max(len(vec) for vec in encoded_texts)
        encoded_matrix = np.zeros((len(encoded_texts), max_length))
        
        for i, vec in enumerate(encoded_texts):
            encoded_matrix[i, :len(vec)] = vec[:max_length]
        
        encoded_matrices.append(encoded_matrix)  # Store encoding for this column
    # Merge all encoded column matrices into one final feature matrix
    final_feature_matrix = np.hstack(encoded_matrices)  # Horizontally stack them

    return final_feature_matrix

def cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4):
    num_clusters = len(df) // min_group_size  # Ensure enough clusters for min size
    
    # Apply K-Means clustering
    from sklearn.cluster import KMeans
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
file_path = 'synthetic_student_availability_open_ended.csv'
df = preprocess_data(file_path)
#feature_matrix = encode_answers(df, 3)
feature_matrix, text_data = create_feature_matrix(df, n=3)  # Generate numerical feature matrix with trigrams
df_clustered = cluster_students(df, feature_matrix, min_group_size=3, max_group_size=4)
generate_report(df_clustered, 'student_group_assignments_v6.csv')

