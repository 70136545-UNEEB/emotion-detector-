import pickle
import os
import sys
import sklearn

# Add path to find modules if needed
sys.path.append(os.getcwd())

print(f"Python executable: {sys.executable}")
print(f"Sklearn version: {sklearn.__version__}")
print(f"Sklearn file: {sklearn.__file__}")

print("Inspect model...")
try:
    with open('model/emotion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print(f"Type: {type(model)}")
    
    if hasattr(model, 'multi_class'):
        print(f"multi_class: {model.multi_class}")
    else:
        print("multi_class attribute is MISSING")
        
    # Try to predict_proba
    try:
        # Create dummy input
        # We need the vectorizer too to get correct shape
        with open('model/vectorizer.pkl', 'rb') as f:
            vec = pickle.load(f)
        
        text = ["happy"]
        X = vec.transform(text)
        print("Predicting proba...")
        prob = model.predict_proba(X)
        print(f"Proba: {prob}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Error: {e}")
