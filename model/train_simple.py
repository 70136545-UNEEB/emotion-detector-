import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Create sample data if files don't exist
def create_sample_data():
    """Create sample data if no dataset files are found"""
    data = {
        'text': [
            'I am happy today this is amazing',
            'I feel really sad and disappointed',
            'This makes me so angry I cant believe it',
            'I love this so much its wonderful',
            'I am scared and worried about what might happen',
            'Wow that was surprising I didnt expect that',
            'Feeling joyful and excited about the future',
            'This is terrible and depressing',
            'I hate when this happens its so frustrating',
            'This is so romantic and beautiful',
            'Im terrified of what comes next',
            'That was completely unexpected amazing',
            'Everything is going great today',
            'I feel so lonely and miserable',
            'This injustice makes me furious',
            'I adore everything about this',
            'Im anxious about the results',
            'What a shocking revelation'
        ],
        'label': [
            'joy', 'sadness', 'anger', 'love', 'fear', 'surprise',
            'joy', 'sadness', 'anger', 'love', 'fear', 'surprise',
            'joy', 'sadness', 'anger', 'love', 'fear', 'surprise'
        ]
    }
    
    df = pd.DataFrame(data)
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/emotions.csv', index=False)
    
    # Also create txt files
    with open('../data/train.txt', 'w') as f:
        for i in range(12):  # First 12 for training
            f.write(f"{data['label'][i]}\t{data['text'][i]}\n")
    
    with open('../data/test.txt', 'w') as f:
        for i in range(12, 15):  # Next 3 for testing
            f.write(f"{data['label'][i]}\t{data['text'][i]}\n")
    
    with open('../data/val.txt', 'w') as f:
        for i in range(15, 18):  # Last 3 for validation
            f.write(f"{data['label'][i]}\t{data['text'][i]}\n")
    
    print("Sample dataset created!")
    return df

# Try to load existing data
def load_data():
    """Try to load data from various sources"""
    
    # Try TXT files first
    def load_txt_file(file_path):
        texts, labels = [], []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        label, text = line.split('\t', 1)
                        texts.append(text.strip())
                        labels.append(label.strip())
            return texts, labels
        except:
            return [], []
    
    # Load from TXT files
    train_texts, train_labels = load_txt_file('../data/train.txt')
    test_texts, test_labels = load_txt_file('../data/test.txt')
    val_texts, val_labels = load_txt_file('../data/val.txt')
    
    all_texts = train_texts + test_texts + val_texts
    all_labels = train_labels + test_labels + val_labels
    
    if all_texts:
        return pd.DataFrame({'text': all_texts, 'label': all_labels})
    
    # Try CSV file
    try:
        df = pd.read_csv('../data/emotions.csv')
        return df
    except:
        pass
    
    # Create sample data as last resort
    return create_sample_data()

# Main training
print("=== Simple Emotion Detection Training ===")

# Load data
df = load_data()
print(f"Dataset loaded: {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Prepare data
X = df['text']
y = df['label']

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
reverse_mapping = {idx: label for label, idx in label_mapping.items()}
y_encoded = np.array([label_mapping[label] for label in y])

print(f"Label mapping: {label_mapping}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
y_test_labels = [reverse_mapping[idx] for idx in y_test]
y_pred_labels = [reverse_mapping[idx] for idx in y_pred]
print(classification_report(y_test_labels, y_pred_labels))

# Save model
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

print("\nModel saved successfully!")