from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import logging
import warnings
from helpers import vn_processing as xt

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MySQL database configuration
db_config = {
    'user': 'root',
    'password': 'root1234',
    'host': 'localhost',
    'database': 'rcm'
}

# Load trained models and vectorizer
with open('models/xgboots_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to save prediction to MySQL
def save_csv_to_mysql(file_path):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Load CSV data into pandas DataFrame
        restaurants = pd.read_csv(file_path)

        # Preprocess data
        restaurants = preprocess_restaurants_data(restaurants)

        # Insert each row into MySQL table
        for _, row in restaurants.iterrows():
            query = """
            INSERT INTO restaurants (RestaurantID, RestaurantName, Address, District, Time, TimeOpen, TimeClose, LowestPrice, HighestPrice) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (row['RestaurantID'], row['Restaurant Name'], row['Address'], row['District'], row['Time'], row['Time Open'], row['Time Close'], row['Lowest Price'], row['Highest Price'])
            cursor.execute(query, values)
        
        conn.commit()
        cursor.close()
        conn.close()

        logging.info("Data successfully saved to MySQL")
        return True
    except mysql.connector.Error as err:
        logging.error(f"Error saving data to MySQL: {err}")
        return False

def preprocess_restaurants_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['Restaurant Name'])
    df['District'] = df['Address'].str.split(', ').apply(lambda x: x[-2] if len(x) > 1 else None)
    df = df.drop(columns='Unnamed: 0')
    df['Time'] = df['Time'].fillna('00:00 - 23:59')
    df['Time Open'] = df['Time'].apply(lambda x: x[:5])
    df['Time Close'] = df['Time'].apply(lambda x: x[-5:])
    df['Lowest Price'] = df['Price'].str.split(' - ').str[0].str.replace(".", "").astype('float')
    df['Highest Price'] = df['Price'].str.split(' - ').str[1].str.replace(".", "").astype('float')
    df.loc[df['Lowest Price'] < 1000, 'Lowest Price'] = 20000
    df.loc[df['Highest Price'] < 1000, 'Highest Price'] = 50000
    df = df.sort_values(by='RestaurantID')
    df['RestaurantID'] = df['RestaurantID'].astype(str)
    df = df.drop(columns='Price')
    return df

# Function to save each comment and its label to MySQL
def save_to_mysql(comment, label):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
        INSERT INTO sentiment_analysis (Comment, Label)
        VALUES (%s, %s)
        """
        values = (comment, label)
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logging.error(f"Error saving sentiment data to MySQL: {err}")

# Function to preprocess text and make predictions
def predict_sentiment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = model.predict(X_test)
    df['Label'] = y_pred
    df['Label'] = df['Label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    df = df[['Comment', 'Label']]
    
    # Save each comment and its label to MySQL
    for _, row in df.iterrows():
        save_to_mysql(row['Comment'], row['Label'])
    
    return df.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json  # Get JSON data from request
        text = content.get('text')  # Access 'text' key from JSON data
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        result = predict_sentiment([text])
        return jsonify(result)
    except KeyError:
        return jsonify({'error': 'Missing required parameter'}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
