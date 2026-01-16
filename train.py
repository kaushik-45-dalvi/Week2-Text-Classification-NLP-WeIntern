import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

# Load dataset
df = pd.read_csv("spam.csv")

# Preprocess text
df['text'] = df['text'].apply(clean_text)

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']  
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model training completed")
