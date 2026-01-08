# PM Internship Recommendation Engine 

An AI-powered web application that recommends internships based on a user's
skills, domain interests, and preferred location.

##  Problem Statement
Students often struggle to find relevant internships due to the vast number
of listings and lack of personalized recommendations.

##  Solution
This system uses Natural Language Processing (NLP) and Machine Learning
techniques to recommend the most relevant internships by comparing user inputs
with internship descriptions.

##  Technologies Used
- Python
- Flask
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- HTML, CSS
- Render (Deployment)

##  How It Works
1. User enters skills, domain, and location
2. Text data is vectorized using TF-IDF
3. Cosine similarity is calculated
4. Top matching internships are displayed

##  Live Demo
  **Deployed on Render:**  
(https://pm-internship-recommendation-engine-ua5n.onrender.com/)

##  Project Structure
pm-internship-recommendation-engine/
│
├── app.py
├── requirements.txt
├── merged_internships_dataset.csv
│
├── templates/
│ ├── index.html
│ └── results.html
│
├── static/
│ └── style.css

##  Future Enhancements
- Improve recommendation ranking
- Add user login
- Cache vectorizer for faster performance
- Improve UI/UX

##  Author
**Jyothirmai Pasupuleti**  
Third Year Student – AI / ML
