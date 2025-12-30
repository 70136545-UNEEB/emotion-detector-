from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import traceback

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__, static_folder='static', template_folder='templates')

class EmotionPredictor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        self.label_mapping = None
        self.reverse_mapping = None
        self.load_model()
    
    def load_model(self):
        try:
            print("DEBUG: Loading emotion model...")
            
            # Check if model files exist
            model_files = ['emotion_model.pkl', 'vectorizer.pkl', 'label_mapping.pkl']
            for file in model_files:
                if not os.path.exists(f'model/{file}'):
                    print(f"ERROR: Model file not found: model/{file}")
                    self.setup_fallback_model()
                    return
            
            with open('model/emotion_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Patch for sklearn compatibility
            if not hasattr(self.model, 'multi_class'):
                print("INFO: Patching model with default multi_class='auto'")
                self.model.multi_class = 'auto'
                
            with open('model/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('model/label_mapping.pkl', 'rb') as f:
                self.label_mapping = pickle.load(f)
            
            self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            print("INFO: Model loaded successfully!")
            print(f"INFO: Available emotions: {list(self.label_mapping.keys())}")
            
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            print("INFO: Setting up fallback model...")
            self.setup_fallback_model()
    
    def setup_fallback_model(self):
        """Create a simple fallback model if main model fails to load"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        print("INFO: Creating fallback model...")
        
        # Simple training data
        texts = [
            'I am happy and excited', 
            'I feel sad and lonely', 
            'This makes me angry', 
            'I love this so much', 
            'I am scared and worried', 
            'This is surprising'
        ]
        labels = ['joy', 'sadness', 'anger', 'love', 'fear', 'surprise']
        
        self.label_mapping = {label: idx for idx, label in enumerate(labels)}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(texts)
        
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, [self.label_mapping[label] for label in labels])
        print("INFO: Fallback model created!")
    
    def clean_text(self, text):
        try:
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Convert to lowercase
            text = text.lower()
            # Tokenize
            tokens = nltk.word_tokenize(text)
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            print(f"ERROR: Text cleaning error: {e}")
            return text.lower()  # Fallback: just lowercase
    
    def predict_emotion(self, text):
        try:
            if not text or len(text.strip()) < 2:
                return {
                    'emotion': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {'unknown': 1.0},
                    'cleaned_text': text
                }
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Vectorize
            text_vec = self.vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            
            # Get emotion label
            emotion = self.reverse_mapping[prediction]
            confidence = probabilities[prediction]
            
            print(f"DEBUG: probabilities length: {len(probabilities)}")
            print(f"DEBUG: reverse_mapping: {self.reverse_mapping}")

            # Get all probabilities
            emotion_probs = {}
            for i, prob in enumerate(probabilities):
                emotion_name = self.reverse_mapping[i]
                emotion_probs[emotion_name] = float(prob)
            
            print(f"INFO: Prediction: '{text[:30]}...' -> {emotion} ({confidence:.2f})")
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'probabilities': emotion_probs,
                'cleaned_text': cleaned_text
            }
            
        except Exception as e:
            error_msg = f"ERROR: Prediction error: {e}\nDETAILS: {traceback.format_exc()}"
            try:
                print(error_msg)
            except:
                pass
            with open('server_error.log', 'a') as f:
                f.write(error_msg + "\n")
            return None

# Initialize predictor
predictor = EmotionPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"INFO: Analyzing text: {text[:50]}...")
        result = predictor.predict_emotion(text)
        
        if result:
            return jsonify({
                'success': True,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'cleaned_text': result['cleaned_text']
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        print(f"ERROR: Error in predict route: {e}")
        print(f"DETAILS: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        print(f"INFO: Analyzing batch of {len(texts)} texts...")
        results = []
        for text in texts:
            if text and text.strip():  # Only process non-empty texts
                result = predictor.predict_emotion(text)
                if result:
                    results.append({
                        'text': text,
                        'emotion': result['emotion'],
                        'confidence': result['confidence']
                    })
        
        print(f"INFO: Batch analysis completed: {len(results)} results")
        return jsonify({
            'success': True,
            'results': results,
            'total_analyzed': len(results)
        })
            
    except Exception as e:
        print(f"ERROR: Error in batch analysis: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/test')
def test():
    """Test route to check if everything is working"""
    return jsonify({
        'status': 'success',
        'message': 'Flask server is running!',
        'model_loaded': predictor.model is not None,
        'emotions_available': list(predictor.label_mapping.keys()) if predictor.label_mapping else []
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'static_files': os.path.exists('static/css/style.css')
    })

if __name__ == '__main__':
    print("INFO: Starting Emotion Detection Server...")
    print("INFO: Static folder:", app.static_folder)
    print("INFO: Template folder:", app.template_folder)
    print("INFO: Checking static files...")
    print("   CSS exists:", os.path.exists('static/css/style.css'))
    print("   JS exists:", os.path.exists('static/js/script.js'))
    app.run(debug=True, host='0.0.0.0', port=5000)