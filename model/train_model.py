import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class EmotionDetector:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        self.label_mapping = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
    
    def clean_text(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path):
        # Load dataset
        df = pd.read_csv(file_path)
        print("Dataset columns:", df.columns)
        print("Sample data:")
        print(df.head())
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Map labels
        df['label'] = df['label'].map(self.label_mapping)
        
        return df
    
    def train_traditional_ml(self, df):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Traditional ML Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_mapping.keys()))
        
        return accuracy
    
    def save_model(self):
        # Save the model and vectorizer
        with open('emotion_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('label_mapping.pkl', 'wb') as f:
            pickle.dump(self.label_mapping, f)
        
        print("Model saved successfully!")

def main():
    detector = EmotionDetector()
    
    # Load and preprocess data
    df = detector.load_and_preprocess_data('../data/emotions.csv')
    
    # Train traditional ML model
    accuracy = detector.train_traditional_ml(df)
    
    # Save model
    detector.save_model()
    
    print(f"Model training completed with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()