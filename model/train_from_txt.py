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
        self.label_mapping = {}
    
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def load_txt_data(self, file_path):
        """Load data from .txt file with tab separation"""
        texts = []
        labels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try tab separation first (most common)
                    if '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            label, text = parts
                            labels.append(label.strip())
                            texts.append(text.strip())
                        else:
                            print(f"Warning: Line {line_num} has unexpected format: {line}")
                    else:
                        # Try other separators
                        for sep in [';', ',']:
                            if sep in line:
                                parts = line.split(sep, 1)
                                if len(parts) == 2:
                                    label, text = parts
                                    labels.append(label.strip())
                                    texts.append(text.strip())
                                break
                        else:
                            # Try space separation as last resort
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                label, text = parts
                                labels.append(label.strip())
                                texts.append(text.strip())
                            else:
                                print(f"Warning: Cannot parse line {line_num}: {line}")
                                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    def load_and_preprocess_data(self):
        """Load and combine train, test, val files"""
        print("Loading dataset files...")
        
        # Load all files
        datasets = {}
        for file_name in ['train.txt', 'test.txt', 'val.txt']:
            file_path = f'../data/{file_name}'
            if os.path.exists(file_path):
                df = self.load_txt_data(file_path)
                datasets[file_name] = df
                print(f"{file_name}: {len(df)} samples")
            else:
                print(f"Warning: {file_path} not found")
        
        if not datasets:
            print("No data files found! Please check your data directory.")
            return None
        
        # Combine all data
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        
        print(f"\nTotal samples: {len(combined_df)}")
        print("\nLabel distribution:")
        print(combined_df['label'].value_counts())
        
        # Create label mapping dynamically
        unique_labels = combined_df['label'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"\nLabel mapping: {self.label_mapping}")
        
        # Clean text
        print("\nCleaning text...")
        combined_df['cleaned_text'] = combined_df['text'].apply(self.clean_text)
        
        # Map labels to numbers
        combined_df['label_num'] = combined_df['label'].map(self.label_mapping)
        
        # Remove any rows with NaN labels
        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['label_num'])
        final_count = len(combined_df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with invalid labels")
        
        print(f"\nFinal dataset size: {len(combined_df)}")
        
        return combined_df
    
    def train_model(self, df):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Vectorize text
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Convert numeric labels back to text for classification report
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        y_test_text = [reverse_mapping[int(x)] for x in y_test]
        y_pred_text = [reverse_mapping[int(x)] for x in y_pred]
        
        print("\nClassification Report:")
        print(classification_report(y_test_text, y_pred_text))
        
        return accuracy
    
    def save_model(self):
        """Save the model and preprocessing objects"""
        # Save the model and vectorizer
        with open('emotion_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('label_mapping.pkl', 'wb') as f:
            pickle.dump(self.label_mapping, f)
        
        print("\nModel saved successfully!")
        print("Files created:")
        print("- emotion_model.pkl")
        print("- vectorizer.pkl") 
        print("- label_mapping.pkl")

def main():
    detector = EmotionDetector()
    
    # Load and preprocess data
    df = detector.load_and_preprocess_data()
    
    if df is None or len(df) == 0:
        print("No data loaded! Please check your data files.")
        return
    
    # Train model
    accuracy = detector.train_model(df)
    
    # Save model
    detector.save_model()
    
    print(f"\n=== Training Completed ===")
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()