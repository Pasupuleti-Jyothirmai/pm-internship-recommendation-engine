from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "pm_internship_secret_key"


# -----------------------------
# LOAD & PREPARE DATA (GLOBAL)
# -----------------------------
df = pd.read_csv("merged_internships_dataset.csv")

df = df[["profile", "Skills", "Location", "Education"]]
df = df.fillna("")

df["combined_text"] = (
    df["profile"] + " " +
    df["Skills"] + " " +
    df["Education"] + " " +
    df["Location"]
)

# -----------------------------
# BUILD TF-IDF MODEL (GLOBAL)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

print("TF-IDF model built successfully")
translations = {
    "en": {
        "language_code": "en",
        "page_title": "PM Internship Recommendation Engine",
        "language": "Language",
        "skills": "Skills",
        "sector": "Sector / Interest",
        "location": "Location",
        "submit": "Get Recommendations",
        "title": "Recommended Internships",
        "role": "Role",
        "education": "Education"
    },
    "hi": {
        "language_code": "hi",
        "page_title": "पीएम इंटर्नशिप अनुशंसा प्रणाली",
        "language": "भाषा",
        "skills": "कौशल",
        "sector": "क्षेत्र / रुचि",
        "location": "स्थान",
        "submit": "अनुशंसाएँ प्राप्त करें",
        "title": "अनुशंसित इंटर्नशिप",
        "role": "भूमिका",
        "education": "शिक्षा"
    },
    "te": {
    "language_code": "te",
    "page_title": "పీఎం ఇంటర్న్‌షిప్ సిఫార్సు వ్యవస్థ",
    "language": "భాష",
    "skills": "నైపుణ్యాలు",
    "sector": "రంగం / ఆసక్తి",
    "location": "స్థానం",
    "submit": "సిఫార్సులు పొందండి",
    "title": "సిఫార్సు చేసిన ఇంటర్న్‌షిప్‌లు",
    "role": "పాత్ర",
    "education": "విద్య"
}


}




# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    if "language" not in session:
        session["language"] = "en"

    language = session.get("language", "en")
    labels = translations[language]

    return render_template(
        "index.html",
        labels=labels,
        language=language
    )



@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return redirect(url_for("home"))

    # ✅ language ALWAYS comes from session
    language = session.get("language", "en")
    labels = translations[language]

    skills = request.form["skills"]
    sector = request.form["sector"]
    location = request.form["location"]

    user_input = skills + " " + sector + " " + location

    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    top_indices = similarity_scores[0].argsort()[-5:][::-1]

    recommendations = []

    for idx in top_indices:
        internship = df.iloc[idx]
        recommendations.append({
            "profile": internship["profile"],
            "skills": internship["Skills"],
            "location": internship["Location"],
            "education": internship["Education"]
        })

    return render_template(
        "results.html",
        recommendations=recommendations,
        labels=labels,
        language=language
    )


@app.route("/set_language", methods=["POST"])
def set_language():
    session["language"] = request.form["language"]
    return redirect(url_for("home"))




# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

