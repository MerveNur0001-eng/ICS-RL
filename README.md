
# Mental Health Sentiment & Status Classifier

This project is a machine learning pipeline for analyzing mental health-related text data. It classifies user-provided statements into:

- **Mental health status** (from your dataset's `status` column),
- **Sentiment** using both VADER (rule-based) and supervised ML methods.

The project includes preprocessing, model training using `LinearSVC`, evaluation, and visualization.

---


## ⚙️ Requirements

- Python 3.7+
- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn
- vaderSentiment

Install required packages:

```bash
pip install pandas nltk scikit-learn matplotlib seaborn vaderSentiment
```

Also, make sure to download the required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 🚀 How to Run

1. Place your dataset as `Combined_Data.csv` in the root directory. It must include:
   - `statement` column (text data)
   - `status` column (label)

2. Run the script:

```bash
python app.py
```

3. The script will:
   - Clean and preprocess the text
   - Apply VADER sentiment analysis
   - Train two models (`status` and `sentiment`)
   - Evaluate their accuracy
   - Save trained models (`.pkl`)
   - Generate a sentiment distribution plot (`static/sentiment_distribution.png`)

---

## 🧠 Sentiment Analysis

This project uses **VADER** for rule-based sentiment scoring and also trains a **LinearSVC** classifier using cleaned text data.

Sentiment labels:
- `Positive`
- `Negative`
- `Neutral`

---

## 📝 Logging & Error Handling

The script includes extensive logging and error checking:
- Logs when files are missing or columns are incorrect
- Reports timing for preprocessing tasks
- Warnings if cleaned statements are empty

---

## 💾 Model Saving

The following models are saved after training:

- `status_pipeline.pkl`
- `sentiment_pipeline.pkl`

You can later load them for predictions using:

```python
import pickle
with open('status_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 🔍 Future Ideas

- Integrate with Flask to build a web app.
- Extend to multilingual sentiment detection.
- Add visual dashboards for user insights.

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out!
