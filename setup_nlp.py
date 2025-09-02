import nltk
import textblob

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Download TextBlob corpora
import subprocess
subprocess.run(["python", "-m", "textblob.download_corpora"])
