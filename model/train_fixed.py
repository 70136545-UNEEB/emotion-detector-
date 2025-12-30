import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("=== Emotion Detection Model Training ===")

def load_txt_files():
    """Load all txt files and combine them"""
    all_texts = []
    all_labels = []
    
    files = ['../data/train.txt', '../data/test.txt', '../data/val.txt']
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and '\t' in line:
                        try:
                            label, text = line.split('\t', 1)
                            all_texts.append(text.strip())
                            all_labels.append(label.strip())
                        except Exception as e:
                            print(f"Error parsing line {line_num} in {file_path}: {e}")
                            continue
    
    print(f"Total samples loaded: {len(all_texts)}")
    return all_texts, all_labels

# Load data
texts, labels = load_txt_files()

if len(texts) == 0:
    print("No data loaded! Using fallback data...")
    # Fallback data
    texts = [
        "I am happy today this is amazing",
        "I feel really sad and disappointed", 
        "This makes me so angry",
        "I love this so much",
        "I am scared and worried",
        "Wow that was surprising"
    ]
    labels = ["joy", "sadness", "anger", "love", "fear", "surprise"]

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

print(f"\nDataset shape: {df.shape}")
print("Label distribution:")
print(df['label'].value_counts())

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
reverse_mapping = {idx: label for label, idx in label_mapping.items()}

print(f"\nLabel mapping: {label_mapping}")

# Prepare features and labels
X = df['text']
y = np.array([label_mapping[label] for label in df['label']])

# Vectorize text
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=1000, 
    stop_words='english',
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

# Train model
print("Training model...")
model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000, 
    random_state=42
)
model.fit(X_vec, y)

# Evaluate
y_pred = model.predict(X_vec)
accuracy = accuracy_score(y, y_pred)

print(f"\n=== Training Results ===")
print(f"Accuracy: {accuracy:.4f}")

# Show detailed report
print("\nClassification Report:")
y_labels = [reverse_mapping[idx] for idx in y]
y_pred_labels = [reverse_mapping[idx] for idx in y_pred]
print(classification_report(y_labels, y_pred_labels))

# Show some predictions
print("\nSample Predictions:")
print("-" * 50)
for i in range(min(5, len(X))):
    actual = y_labels[i]
    predicted = y_pred_labels[i]
    confidence = max(model.predict_proba(X_vec[i])[0])
    print(f"Text: '{X.iloc[i][:40]}...'")
    print(f"Actual: {actual:8} | Predicted: {predicted:8} | Confidence: {confidence:.2f}")
    print()

# Save model
print("Saving model...")
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

print("\n✅ Model saved successfully!")
print("📁 Files created:")
print("   - emotion_model.pkl")
print("   - vectorizer.pkl") 
print("   - label_mapping.pkl")
print(f"\n🎯 Model can detect {len(label_mapping)} emotions: {list(label_mapping.keys())}")