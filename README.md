# 🛡️ Hate Speech Detection using Machine Learning

This project is a Machine Learning-based text classification system that detects whether a tweet contains hate speech, offensive language, or neither. It uses NLP techniques and a Decision Tree Classifier to analyze social media text and classify it into three categories.

## 📌 Table of Contents
- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Project Pipeline](#-project-pipeline)
- [How to Run](#-how-to-run)
- [Sample Output](#-sample-output)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

## 🧠 Overview

The system classifies tweets into three categories:
- `Hate Speech`
- `Offensive Language`
- `No Hate or Offensive Language`

We preprocess the tweets using Natural Language Processing (NLP), convert them into numerical features, and train a Decision Tree model to classify them accurately.

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- scikit-learn
- Matplotlib, Seaborn

## 🧪 Project Pipeline

1. **Data Loading** – CSV file with labeled tweets
2. **Text Cleaning** – Lowercasing, removing punctuation, links, stopwords, stemming
3. **Feature Extraction** – CountVectorizer to convert text to numerical format
4. **Train-Test Split** – Using `train_test_split()`
5. **Model Training** – Using `DecisionTreeClassifier`
6. **Evaluation** – Confusion Matrix, Accuracy Score
7. **Prediction** – Try on your own sample tweet!

## ▶️ How to Run

1. Clone the repo:
   
   git clone https://github.com/Aidaelizabath/Hate_Speech_Detection.git
   cd Hate_Speech_Detection

2. Install dependencies:

   pip install -r requirements.txt

3.Run the Python script:

   python main.py

4.Output will show accuracy, confusion matrix, and prediction on sample tweet.

   Prediction for sample: Offensive Language
   Confusion Matrix:
   [[175  50 240]
   [ 21 1230 128]
   [267 180 5888]]

Accuracy: 0.84

🌟 Results
✅ Achieved over 84% accuracy on the test set

✅ Cleaned over 20K+ tweets using NLP

✅ Successfully classifies offensive or harmful language in real-time

👨‍💻 Author
Email:aidaelizabathvarghese2003@gmail.com

LinkedIn: https://www.linkedin.com/in/aida-elizabath-varghese-ab4b9b34b/

GitHub: https://github.com/Aidaelizabath
