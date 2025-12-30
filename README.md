# 🎭 Emotion Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

A powerful real-time text emotion recognition system built with **Flask**, **Scikit-learn**, and **NLTK**.  
This application analyzes text input to predict underlying emotions (**Joy, Sadness, Anger, Fear, Love, Surprise**) with high accuracy.

---

## 🚀 Features

*   **Real-time Analysis**: Instant emotion prediction for single sentences.
*   **Batch Processing**: Analyze multiple texts simultaneously via API.
*   **Visual Feedback**: Clean, modern UI with confidence scores and probability distributions.
*   **Robust Model**: Trained on 20,000+ samples using Logistic Regression (~89% accuracy).
*   **REST API**: JSON endpoints for easy integration with other services.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask
*   **ML Engine**: Scikit-learn, NLTK (WordNet, Stopwords), TF-IDF
*   **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
*   **Deployment**: Ready for Gunicorn / Heroku / Render

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/70136545-UNEEB/emotion-detector-.git
    cd emotion-detector-
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv emotion_env
    
    # Windows
    emotion_env\Scripts\activate
    
    # Mac/Linux
    source emotion_env/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Run the Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
👉 **http://127.0.0.1:5000**

### 2. API Endpoints

#### `POST /predict`
Analyze a single text string.

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"I am so happy today!\"}" http://127.0.0.1:5000/predict
```

**Response:**
```json
{
  "emotion": "joy",
  "confidence": 0.98,
  "probabilities": { ... },
  "success": true
}
```

#### `POST /analyze_batch`
Analyze a list of texts.

**Request:**
```json
{
  "texts": ["I am sad", "This is amazing!"]
}
```

## 📂 Project Structure

```bash
emotion-detector/
├── app.py                 # Main Flask application entry point
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment configuration
│
├── model/                 # ML Model artifacts & training scripts
│   ├── emotion_model.pkl  # Trained Logistic Regression model
│   ├── vectorizer.pkl     # TF-IDF Vectorizer
│   ├── label_mapping.pkl  # Emotion label mapping
│   └── train_direct.py    # Script used to train the model
│
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Custom styling
│   └── js/
│       └── script.js      # Frontend logic
│
├── templates/
│   └── index.html         # Main web interface
│
└── data/                  # Dataset directory
    ├── train.txt
    ├── test.txt
    └── val.txt
```

## 📊 Model Performance

The model utilizes **Logistic Regression** with **TF-IDF Vectorization** trained on a dataset of 20,000 labeled tweets.

*   **Accuracy**: ~88.7%
*   **Classes**: Joy, Sadness, Anger, Fear, Love, Surprise
*   **Preprocessing**: 
    *   Text cleaning (Regex)
    *   Tokenization
    *   Stopword removal (NLTK)
    *   Lemmatization (WordNet)

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Made with ❤️ by [Uneeb Ali](https://github.com/70136545-UNEEB)
