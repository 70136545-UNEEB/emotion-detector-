import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

print("=== Emotion Detection Model Training ===")

def load_data_direct():
    """Load data directly from txt files"""
    texts = []
    labels = []
    
    files = [
        '../data/train.txt',
        '../data/test.txt', 
        '../data/val.txt'
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"📖 Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            if ';' in line:
                                text, label = line.split(';', 1)
                                texts.append(text.strip())
                                labels.append(label.strip())
                            elif '\t' in line:
                                label, text = line.split('\t', 1)
                                texts.append(text.strip())
                                labels.append(label.strip())
                        except Exception as e:
                            print(f"⚠️ Error parsing line {line_num}: {e}")
                            continue
        else:
            print(f"❌ File not found: {file_path}")
    
    return texts, labels

# Load data
print("🔄 Loading dataset...")
texts, labels = load_data_direct()

if len(texts) == 0:
    print("❌ No data loaded! Using fallback data...")
    # Fallback data
    texts = [
        "I am happy today this is amazing",
        "I feel really sad and disappointed",
        "This makes me so angry I cant believe it",
        "I love this so much its wonderful",
        "I am scared and worried about what might happen",
        "Wow that was surprising I didnt expect that"
    ]
    labels = ["joy", "sadness", "anger", "love", "fear", "surprise"]

print(f"✅ Loaded {len(texts)} samples")

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})
print(f"📊 Dataset shape: {df.shape}")

print("📈 Label distribution:")
label_counts = df['label'].value_counts()
print(label_counts)

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
reverse_mapping = {idx: label for label, idx in label_mapping.items()}

print(f"\n🎯 Label mapping: {label_mapping}")

# Prepare features and labels
X = df['text']
y = np.array([label_mapping[label] for label in df['label']])

# Vectorize text
print("\n🔧 Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=1000, 
    stop_words='english',
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

# Train model
print("🤖 Training model...")
model = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    C=1.0
)
model.fit(X_vec, y)

# Evaluate
y_pred = model.predict(X_vec)
accuracy = accuracy_score(y, y_pred)

print(f"\n=== 📊 Training Results ===")
print(f"✅ Accuracy: {accuracy:.4f}")

# Show detailed report
print("\n📋 Classification Report:")
y_labels = [reverse_mapping[idx] for idx in y]
y_pred_labels = [reverse_mapping[idx] for idx in y_pred]
print(classification_report(y_labels, y_pred_labels))

# Show some predictions
print("\n🎭 Sample Predictions:")
print("-" * 60)
for i in range(min(6, len(X))):
    actual = y_labels[i]
    predicted = y_pred_labels[i]
    confidence = max(model.predict_proba(X_vec[i])[0])
    status = "✅" if actual == predicted else "❌"
    print(f"{status} Text: '{X.iloc[i][:50]}...'")
    print(f"   Actual: {actual:8} | Predicted: {predicted:8} | Confidence: {confidence:.2f}")

# Save model
print("\n💾 Saving model...")
with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

print("\n🎉 Model saved successfully!")
print("📁 Files created in model directory:")
print("   - emotion_model.pkl")
print("   - vectorizer.pkl") 
print("   - label_mapping.pkl")
print(f"\n🔮 Model can detect {len(label_mapping)} emotions: {list(label_mapping.keys())}")

# Test the model
print("\n🧪 Testing saved model...")
try:
    with open('emotion_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    with open('label_mapping.pkl', 'rb') as f:
        loaded_mapping = pickle.load(f)
    
    print("✅ Model files verified and ready to use!")
    
    # Quick test prediction
    test_text = "I am feeling happy and excited"
    test_vec = loaded_vectorizer.transform([test_text])
    test_pred = loaded_model.predict(test_vec)[0]
    test_emotion = reverse_mapping[test_pred]
    print(f"🧪 Test prediction: '{test_text}' -> {test_emotion}")
    
except Exception as e:
    print(f"❌ Error testing saved model: {e}")