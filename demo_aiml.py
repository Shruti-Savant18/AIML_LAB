import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('news.csv')

# Data Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)

# Features and Labels
X = data['text']
y = data['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Model Training using Naive Bayes
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Model Evaluation
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Accuracy
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

# Function to predict if a news article is fake or real
def predict_news(text):
    sample_text_preprocessed = preprocess_text(text)
    sample_text_counts = vectorizer.transform([sample_text_preprocessed])
    prediction = model.predict(sample_text_counts)
    return prediction[0]

# Example usage
while True:
    user_input = input("Enter news text (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = predict_news(user_input)
    print(f"Prediction for the entered news: {prediction}")
