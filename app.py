
from fastapi import BackgroundTasks
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import subprocess
import threading

app = FastAPI(title="InternShipe-Data Analytics-Task4 : Anime Sentiment App (Frontend + API)")
BASE_DIR = Path(__file__).resolve().parent  
DATA_DIR = BASE_DIR / "data"

# serve static and templates
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("templates"):
    os.makedirs("templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

refresh_status = {"running": False, "finished": False, "error": None}

def run_prepare_data():
    global refresh_status
    refresh_status = {"running": True, "finished": False, "error": None}
    try:
        subprocess.run(["python", str(BASE_DIR / "prepare_data.py")], check=True)
        refresh_status = {"running": False, "finished": True, "error": None}
    except subprocess.CalledProcessError as e:
        refresh_status = {"running": False, "finished": True, "error": str(e)}

# --------- Helpers ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure dataframe has no NaN/Inf before sending JSON"""
    return (
        df.fillna("Unknown")
        .replace([float("inf"), float("-inf")], 0)
    )

def load_data():
    try:
        df = pd.read_csv(DATA_DIR/"with_emotions.csv")  
    except Exception:
        df = pd.read_csv(DATA_DIR/"with_sentiment.csv")
    return clean_df(df)

def load_genres():
    try:
        g = pd.read_csv(DATA_DIR/"anime_with_sentiment_genres.csv")
        return clean_df(g)
    except Exception:
        try:
            a = pd.read_csv(DATA_DIR/"anime.csv")
            g = a.dropna(subset=['genre']).assign(
                genre=a['genre'].str.split(', ')
            ).explode('genre')
            g = clean_df(g)
            g.to_csv(DATA_DIR/"anime_with_sentiment_genres.csv", index=False)
            return g
        except Exception:
            return pd.DataFrame()


# --------- Template Routes ----------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "InternShipe-Data Analytics-Task4 : Classification"})

@app.get("/emotions", response_class=HTMLResponse)
def emotions(request: Request):
    return templates.TemplateResponse("emotions.html", {"request": request, "title": "InternShipe-Data Analytics-Task4 : Emotions"})

@app.get("/trends", response_class=HTMLResponse)
def trends(request: Request):
    return templates.TemplateResponse("trends.html", {"request": request, "title": "InternShipe-Data Analytics-Task4 : Trends"})

@app.get("/insights", response_class=HTMLResponse)
def insights(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request, "title": "InternShipe-Data Analytics-Task4 : Insights"})


# --------- API Endpoints ----------
@app.get("/api/sentiments")
def api_sentiments(
    limit: int = 100000,
    sentiment: str | None = Query(None),
    min_rating: float = 0,
    max_rating: float = 10,
    anime_type: str | None = Query(None),
    genre: str | None = Query(None)
):
    df = load_data()

    # filters
    if "rating" in df.columns:
        df = df[(df["rating"] != "Unknown")]  # filter out non-numeric first
        df = df[(df["rating"].astype(float) >= min_rating) & (df["rating"].astype(float) <= max_rating)]
    if sentiment:
        df = df[df["sentiment"] == sentiment]
    if anime_type and "type" in df.columns:
        df = df[df["type"].astype(str) == anime_type]
    if genre:
        genres = load_genres()
        if not genres.empty and "anime_id" in df.columns:
            ids = genres[genres["genre"] == genre]["anime_id"].unique()
            df = df[df["anime_id"].isin(ids)]

    return JSONResponse(clean_df(df.head(limit)).to_dict(orient="records"))


@app.get("/api/sentiment-distribution")
def api_sentiment_distribution():
    df = load_data()
    vc = df["sentiment"].value_counts().reset_index()
    vc.columns = ["sentiment", "count"]
    return JSONResponse(clean_df(vc).to_dict(orient="records"))


@app.get("/api/emotion-distribution")
def api_emotion_distribution():
    df = load_data()
    if "emotion" not in df.columns:
        return JSONResponse([])
    vc = df["emotion"].value_counts().reset_index()
    vc.columns = ["emotion", "count"]
    return JSONResponse(clean_df(vc).to_dict(orient="records"))


@app.get("/api/genres-top")
def api_genres_top(limit: int = 50):
    genres = load_genres()
    if genres.empty:
        return JSONResponse([])
    vc = genres["genre"].value_counts().head(limit).reset_index()
    vc.columns = ["genre", "count"]
    return JSONResponse(clean_df(vc).to_dict(orient="records"))


@app.get("/api/opinion-trends")
def api_opinion_trends():
    try:
        df = pd.read_csv(DATA_DIR/"opinion_trends.csv")
        return JSONResponse(clean_df(df).to_dict(orient="records"))
    except Exception:
        return JSONResponse([])


@app.get("/api/emotion-trends")
def api_emotion_trends():
    try:
        df = pd.read_csv(DATA_DIR/"emotion_trends.csv")
        return JSONResponse(clean_df(df).to_dict(orient="records"))
    except Exception:
        return JSONResponse([])




@app.get("/api/insights")
def api_insights():
    try:
        text = open(DATA_DIR/"insights_report.md", "r", encoding="utf-8").read()
    except Exception:
        text = "Insights not yet generated. Run the pipeline."

    try:
        df = load_data()
    except Exception as e:
        return {
            "markdown": text,
            "error": f"Failed to load dataset: {str(e)}"
        }

    stats = {}
    sentiments, emotions, genre_counts = {}, {}, {}

    # total & rating
    stats["total_anime"] = int(len(df))
    if "rating" in df.columns:
        try:
            stats["avg_rating"] = float(pd.to_numeric(df["rating"], errors="coerce").mean())
        except Exception:
            stats["avg_rating"] = None

    # sentiments
    if "sentiment" in df.columns:
        sentiments = df["sentiment"].value_counts().to_dict()
        stats["sentiment_distribution"] = (
            df["sentiment"].value_counts(normalize=True).to_dict()
        )
    else:
        stats["sentiment_distribution"] = {}

    # emotions
    if "emotion" in df.columns:
        emotions = df["emotion"].value_counts().to_dict()

    # genres
    genres = load_genres()
    if not genres.empty:
        genre_counts = genres["genre"].value_counts().head(10).to_dict()
        stats["top_genre"] = genres["genre"].value_counts().idxmax()
    else:
        stats["top_genre"] = None

    return {
        "markdown": text,
        "last_updated": os.path.getmtime(DATA_DIR / "insights_report.md") if os.path.exists(DATA_DIR / "insights_report.md") else None,
        "stats": stats,
        "sentiments": sentiments,
        "emotions": emotions,
        "genres": genre_counts,
    }

@app.get("/api/genres-top")
def api_genres_top(limit: int = 50):
    genres = load_genres()
    if genres.empty:
        return JSONResponse([])
    vc = genres["genre"].value_counts().head(limit).reset_index()
    vc.columns = ["genre", "count"]
    return JSONResponse(clean_df(vc).to_dict(orient="records"))

@app.post("/api/refresh-data")
def refresh_data(background_tasks: BackgroundTasks):
    """Kick off background refresh job."""
    if refresh_status["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(run_prepare_data)
    return {"status": "started"}

@app.get("/api/refresh-status")
def refresh_status_endpoint():
    """Check current refresh status."""
    return refresh_status 

