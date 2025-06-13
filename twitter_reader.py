import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')


dataset = pd.read_csv("tweet1.csv")


print(" Null values in each column:")
print(dataset.isnull().sum())
print(dataset.info())
print(dataset.describe())


dataset["labels"] = dataset["class"].map({
    0: "Hate Speech",
    1: "Offensive Language",
    2: "No Hate or Offensive language"
})
print(dataset["labels"])


data = dataset[["tweet", "labels"]]
print(data)


stop_words = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

data["tweet"] = data["tweet"].apply(clean_data)
print(data)


X = np.array(data["tweet"])
Y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


y_pred = dt.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))


sample = "you are a bitch"
sample = clean_data(sample)
sample_transformed = cv.transform([sample])
prediction = dt.predict(sample_transformed)
print("Prediction for sample:", prediction[0])

