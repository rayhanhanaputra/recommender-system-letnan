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
# df, dfUji = train_test_split(df_gabungan, test_size=0.2, random_state=42)

# Preprocessing
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopwordind = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopwordind)
    return text

df['judul_clean'] = df['judul'].apply(clean_text)
df.set_index('judul', inplace=True)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['judul_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_dosen(judul_input):
    judul_tfidf = tfidf_vectorizer.transform([clean_text(judul_input)])
    cosine_scores = cosine_similarity(judul_tfidf, tfidf_matrix)
    dosen_index = cosine_scores.argsort()[0][-1]
    recommended_dosen = df['dosen'].iloc[dosen_index]
    return recommended_dosen

# API endpoint
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    judul_input = data['judul']
    recommended_dosen = recommend_dosen(judul_input)
    return jsonify({'judul':judul_input,'recommended_dosen': recommended_dosen})

if __name__ == '__main__':
    app.run(debug=True)
