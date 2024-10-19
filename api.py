from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import nltk
import pandas as pd
import logging
import warnings
from helpers import vn_processing as xt  # Assuming vn_processing contains stepByStep

warnings.filterwarnings('ignore')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Download the required NLTK resource
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained models and vectorizer
with open('models/voting_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to preprocess text and make predictions
def predict_sentiment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)  # Preprocess comments

    # Transform the comments to TF-IDF
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = model.predict(X_test)

    # Map predictions to labels
    df['Label'] = y_pred
    df['Label'] = df['Label'].map({
        0: 'Tiêu cực',
        1: 'Bình thường',
        2: 'Tích cực',
        3: 'Rất Tiêu cực',
        4: 'Rất Tích cực'
    })
    return df[['Comment', 'Label']].to_dict(orient='records')

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html', result=None)

# Route to handle sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json  # Get JSON data from request
        text = content.get('text')  # Access 'text' key from JSON data

        if not text:  # Check if text is empty
            return jsonify({'error': 'Empty text provided'}), 400

        result = predict_sentiment([text])
        return jsonify(result)

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        return jsonify({'error': 'Invalid input format'}), 400

    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run()
