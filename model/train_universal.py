import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def load_emotion_dataset():
    """Universal loader for emotion dataset txt files"""
    
    def load_file(file_path):
        texts, labels = [], []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        # Try common separators
                        if '\t' in line:
                            parts = line.split('\t', 1)
                        elif ';' in line:
                            parts = line.split(';', 1)
                        elif ',' in line:
                            parts = line.split(',', 1)
                        else:
                            parts = line.split(' ', 1)
                        
                        if len(parts) == 2:
                            label, text = parts[0].strip(), parts[1].strip()
                            texts.append(text)
                            labels.append(label)
                        else:
                            print(f"Warning: Cannot parse line {line_num}: {line}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        return texts, labels
    
    # Load all files
    print("Loading dataset files...")
    
    train_texts, train_labels = load_file('../data/train.txt')
    test_texts, test_labels = load_file('../data/test.txt')
    val_texts, val_labels = load_file('../data/val.txt')
    
    print(f"Loaded: {len(train_texts)} train, {len(test_texts)} test, {len(val_texts)} val samples")
    
    # Combine for training
    all_texts = train_texts + test_texts + val_texts
    all_labels = train_labels + test_labels + val_labels
    
    if len(all_texts) == 0:
        print("No data loaded! Creating sample data...")
        # Create sample data as fallback
        all_texts = ['I am happy', 'I am sad', 'I am angry', 'I love this', 'I am scared', 'I am surprised']
        all_labels = ['joy', 'sadness', 'anger', 'love', 'fear', 'surprise']
    
    return pd.DataFrame({'text': all_texts, 'label': all_labels})

# Main training
print("=== Emotion Detection Model Training ===")
df = load_emotion_dataset()

print(f"\nDataset shape: {df.shape}")
print("Label distribution:")
print(df['label'].value_counts())

# Prepare data
X = df['text']
y = df['label']

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
reverse_mapping = {idx: label for label, idx in label_mapping.items()}
y_encoded = np.array([label_mapping[label] for label in y])

print(f"\nLabel mapping: {label_mapping}")

# Vectorize
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_vec, y_encoded)

# Evaluate
y_pred = model.predict(X_vec)
accuracy = accuracy_score(y_encoded, y_pred)

print(f"\n=== Training Results ===")
print(f"Accuracy: {accuracy:.4f}")

# Show sample predictions
print("\nSample predictions:")
for i in range(min(5, len(X))):
    actual = reverse_mapping[y_encoded[i]]
    predicted = reverse_mapping[y_pred[i]]
    print(f"  Text: '{X.iloc[i][:50]}...'")
    print(f"  Actual: {actual}, Predicted: {predicted}")

# Save model
print("\nSaving model...")
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

print("\nModel saved successfully!")
print("Files created in model directory:")
print("- emotion_model.pkl")
print("- vectorizer.pkl")
print("- label_mapping.pkl")