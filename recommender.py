from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import nltk

app = Flask(__name__)

# Load data
df = pd.read_csv("df_gabungan.csv")

# Preprocessing
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopwordind = set(stopwords.words('indonesian'))
stopwordeng = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopwordind)
    text = ' '.join(word for word in text.split() if word not in stopwordeng)
    return text

df['judul_clean'] = df['judul'].apply(clean_text)
df.set_index('judul', inplace=True)

# get most occurence keyword
grouped_by_dosen = df.groupby('dosen')
most_common_keywords = {}

for dosen, group_df in grouped_by_dosen:
    all_titles = ' '.join(group_df['judul_clean'])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([all_titles])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    feature_tfidf_dict = dict(zip(feature_names, tfidf_scores))
    sorted_features = sorted(feature_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, _ in sorted_features[:5]] #change jumlah keyword
    most_common_keywords[dosen] = top_keywords

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['judul_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_dosen(judul_input, n=3): #change jumlah rekomendasi dosen
    judul_tfidf = tfidf_vectorizer.transform([clean_text(judul_input)])
    cosine_scores = cosine_similarity(judul_tfidf, tfidf_matrix)
    top_indices = cosine_scores.argsort()[0][-10:][::-1]
    recommended_dosen = []
    most_occuring_keywords = []

    for idx in top_indices:
        rekomendasi_dosen = df['dosen'].iloc[idx]

        if rekomendasi_dosen not in recommended_dosen:
            recommended_dosen.append(rekomendasi_dosen)
            top_keywords_dosen = most_common_keywords[rekomendasi_dosen]
            most_occuring_keywords.append(top_keywords_dosen)
        
        if len(recommended_dosen) == n:
            break

    return recommended_dosen, most_occuring_keywords

# API endpoint
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    judul_input = data['judul']
    recommended_dosen, most_occuring_keywords = recommend_dosen(judul_input)
    return jsonify({'judul': judul_input, 'recommended_dosen': recommended_dosen, 'most_occuring_keywords': most_occuring_keywords})

if __name__ == '__main__':
    app.run(debug=True)
