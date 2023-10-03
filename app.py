from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)


merged_df = pd.read_csv('data.csv')

# Create TF-IDF vectorizer and compute cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(merged_df.index, index=merged_df['title'])

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None

    if request.method == 'POST':
        game_name = request.form['game_name']
        recommendations = get_recommendations(game_name)

    return render_template('index.html', recommendations=recommendations)


# Function to get game recommendations 
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx])) #similarity scores of all games with that game
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sort the games based on the similarity scores
    sim_scores = sim_scores[1:11]  # Top 10 similar games (excluding the input game)
    game_indices = [i[0] for i in sim_scores]
    recommendations = merged_df.iloc[game_indices][['title','genre','score', 'score_phrase','release_date','platform', 'url']]
    recommendations['Number of Recommendations'] = range(1,len(recommendations)+1)
    
    desired_column_order = [
    "SL NO",
    "Game Names",
    "Review",
    "URL",
    "Platform",
    "Score",
    "Genre",
    "Release Date"
    ]
    recommendations = recommendations.rename(columns={
    "Number of Recommendations": "SL NO",
    "title": "Game Names",
    "score_phrase": "Review",
    "url": "URL",
    "platform": "Platform",
    "score": "Score",
    "genre": "Genre",
    "release_date": "Release Date"
    })[desired_column_order]
    recommendations = recommendations.set_index("SL NO")
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)