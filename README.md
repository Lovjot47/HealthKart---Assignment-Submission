📌 HealthKart – Sentiment Analysis & Brand Recommendation System
📖 Project Overview
This project implements an end-to-end Natural Language Processing (NLP) pipeline to analyze customer product reviews and generate brand recommendations based on sentiment and brand affinity.

The system processes product review text, predicts sentiment using NLP techniques, evaluates model performance, and recommends brands based on category and sentiment-driven affinity scoring.

🎯 Objectives
Perform sentiment analysis on customer reviews

Compare predicted sentiment with rating-based ground truth

Compute brand affinity scores

Generate category-based brand recommendations

Build a modular and production-style ML pipeline

🏗 Tech Stack
Programming Language

Python 3

Libraries

pandas

numpy

nltk

vaderSentiment

scikit-learn

📂 Project Structure
healthkart-assignment/
│
├── data/
│   └── (Dataset should be placed here manually)
│
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   └── sentiment.py
│
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Pipeline Architecture
1️⃣ Data Loading
Loads product review dataset

Performs initial dataset inspection

2️⃣ Text Preprocessing
Lowercasing

Removing special characters

Stopword removal

Lemmatization

3️⃣ Sentiment Analysis
Uses VADER sentiment analyzer

Generates:

Sentiment score

Predicted sentiment label

4️⃣ Ground Truth Sentiment Mapping
Maps ratings into sentiment classes:

4–5 → Positive

3 → Neutral

1–2 → Negative

5️⃣ Model Evaluation
Evaluates prediction performance using:

Accuracy

Precision

Recall

F1 Score

6️⃣ Brand Affinity Scoring
Affinity Score =
(0.7 × Average Sentiment Score) +
(0.3 × Normalized Review Volume)

7️⃣ Recommendation Engine
Recommends top brands based on:

Product category match

Brand affinity score

🚀 How To Run Locally
Step 1 — Clone Repository
git clone https://github.com/Lovjot47/HealthKart---Assignment-Submission.git
cd HealthKart---Assignment-Submission
Step 2 — Create Virtual Environment
python -m venv venv
Activate:

venv\Scripts\activate
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Download NLP Resources
python -m nltk.downloader stopwords punkt wordnet
Step 5 — Add Dataset
Place dataset file inside:

data/GrammarandProductReviews.csv
Step 6 — Run Pipeline
python src/main.py
📊 Outputs Generated
The pipeline generates:

Processed review dataset (in memory or optional save)

Brand affinity scoring table

Sentiment evaluation metrics (console output)

📈 Evaluation Metrics Used
Accuracy

Precision

Recall

F1 Score

🧠 Recommendation Logic
Recommendations are generated using:

✔ Category filtering
✔ Brand sentiment strength
✔ Review volume weighting
✔ Affinity score ranking

⚠ Dataset Note
The original dataset is not included in this repository due to GitHub file size limitations.

To run the project:

Download dataset separately

Place inside data/ folder

Run pipeline normally

🔮 Future Improvements
Transformer-based sentiment models (BERT / RoBERTa)

User-level personalization

Real-time API deployment

Cloud deployment pipeline

Stream processing for real-time reviews

👨‍💻 Author
Lovjot Singh
