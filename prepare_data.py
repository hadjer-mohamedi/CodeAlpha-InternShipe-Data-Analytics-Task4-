import pandas as pd
import kagglehub
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from pathlib import Path
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

BASE_DIR = Path(__file__).resolve().parent  
DATA_DIR = BASE_DIR / "data"
# ========== Load Kaggle Anime Dataset ==========
print("Downloading dataset...")
path = kagglehub.dataset_download("CooperUnion/anime-recommendations-database")
anime_path = os.path.join(path, "anime.csv")
print("anime_path  ",anime_path)
anime = pd.read_csv(anime_path)
print("Anime dataset loaded ✅")

os.makedirs("data", exist_ok=True)
anime.to_csv(DATA_DIR/"anime.csv", index=False)

# ========== Task 1 - Sentiment Classification ==========
def rating_to_sentiment(r):
    print("rating_to_sentiment ",r)
    returnvalue="Neutral"
    try:
        if r >= 7: returnvalue= "Positive"
        elif r >= 4: returnvalue= "Neutral"
        else: returnvalue= "Negative"
    except:
        returnvalue= "Neutral"
    print("rating_to_sentiment ",r)
    print("rating_to_sentiment returnvalue ",returnvalue)    
    return returnvalue    
    #try:
    #    if r >= 7: return "Positive"
    #    elif r >= 4: return "Neutral"
    #    else: return "Negative"
    #except:
    #    return "Neutral"
print('anime["name"] ',anime["name"])
anime["sentiment"] = anime["rating"].apply(rating_to_sentiment)
anime.to_csv(DATA_DIR/"with_sentiment.csv", index=False)
print("Task 1 complete → ",DATA_DIR/"with_sentiment.csv")

# ==========  Task 2 - Emotion Detection ==========
sia = SentimentIntensityAnalyzer()

def detect_emotion(text):
    if not isinstance(text, str):
        return "neutral"
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.6:
        return "joy"
    elif scores["compound"] <= -0.6:
        return "anger"
    elif scores["neg"] > 0.4:
        return "sadness"
    elif scores["pos"] > 0.4:
        return "trust"
    else:
        return "neutral"

anime["emotion"] = anime["name"].apply(detect_emotion)
anime.to_csv(DATA_DIR/"with_emotions.csv", index=False)
print("Task 2 complete → ",DATA_DIR/"with_emotions.csv")

# ==========  Task 3 - Unified Dataset ==========

unified_raw = anime[["anime_id", "name", "genre", "type", "rating", "sentiment", "emotion"]]
unified_raw.to_csv(DATA_DIR/"unified_raw.csv", index=False)
print("Task 3 complete → ",DATA_DIR/"data/unified_raw.csv")

# ========== Task 4 - Opinion Trends ==========

anime_genres = anime.dropna(subset=['genre']).assign(
    genre=anime['genre'].str.split(', ')
).explode('genre')

opinion_trends = anime_genres.groupby(["genre", "sentiment"]).size().unstack(fill_value=0)
opinion_trends.to_csv(DATA_DIR/"opinion_trends.csv")
print("Task 4 complete → ",DATA_DIR/"opinion_trends.csv")
emotion_trends = anime_genres.groupby(["genre", "emotion"]).size().unstack(fill_value=0)
emotion_trends.to_csv(DATA_DIR/"emotion_trends.csv")
print("Task 4b complete → ", DATA_DIR/"emotion_trends.csv")

# ========== Task 5 - Insights Report ==========
total_anime = len(anime)
avg_rating = anime["rating"].mean()


anime_genres = anime.dropna(subset=['genre']).assign(
    genre=anime['genre'].str.split(', ')
).explode('genre')
top_genres = anime_genres["genre"].value_counts().head(5).to_dict()

sentiment_distribution = anime["sentiment"].value_counts(normalize=True) * 100
dominant_sentiment = sentiment_distribution.idxmax()
dominant_pct = sentiment_distribution.max()
sentiment_insight = f"- **{dominant_sentiment}** dominates ({dominant_pct:.1f}%), showing how fans generally feel."

opinion_trends = anime_genres.groupby(["genre", "sentiment"]).size().unstack(fill_value=0)
genre_sentiment_ratios = opinion_trends.div(opinion_trends.sum(axis=1), axis=0)
if "Negative" in genre_sentiment_ratios and "Positive" in genre_sentiment_ratios:
    most_negative_genre = genre_sentiment_ratios["Negative"].idxmax()
    most_positive_genre = genre_sentiment_ratios["Positive"].idxmax()
    genre_insight = f"- **{most_positive_genre}** tends to have the most positive sentiment, while **{most_negative_genre}** leans negative."
else:
    genre_insight = "- Not enough data to compare positive/negative genres."

emotion_trends = anime_genres.groupby(["genre", "emotion"]).size().unstack(fill_value=0)
emotion_totals = emotion_trends.sum().sort_values(ascending=False)
if not emotion_totals.empty:
    top_emotion = emotion_totals.index[0]
    emotion_insight = f"- The leading emotion across anime titles is **{top_emotion}**, shaping audience expectations."
else:
    emotion_insight = "- No strong emotional signals detected in the dataset."

if "type" in anime.columns:
    avg_by_type = anime.groupby("type")["rating"].mean().sort_values(ascending=False)
    if not avg_by_type.empty:
        best_type = avg_by_type.index[0]
        best_type_rating = avg_by_type.iloc[0]
        type_insight = f"- On average, **{best_type}** are rated highest ({best_type_rating:.2f})."
    else:
        type_insight = "- No type-level insights available."
else:
    type_insight = "- Anime format (type) not available."

# Build dynamic report
report = f"""
{sentiment_insight}
{genre_insight}
{emotion_insight}
{type_insight}
"""

# Save report
with open(DATA_DIR/"insights_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Task 5 complete →", DATA_DIR/"insights_report.md")
