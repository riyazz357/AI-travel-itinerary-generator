import os
import json
import pickle
import random
import datetime
import requests
import bcrypt
import mistune  # For parsing AI markdown
import nltk
import numpy as np

# Flask Imports
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

# Import your AI logic
import bard 

from dotenv import load_dotenv

# --- 1. Configuration & Setup ---
load_dotenv()
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")

app = Flask(__name__)
# Enable CORS: This is crucial for React to connect to Flask
CORS(app, resources={r"/*": {"origins": "*"}}) 

app.secret_key = SECRET_KEY or "dev_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- 2. Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    bio = db.Column(db.String(200)) # Added bio field

    def __init__(self, name, email, password, bio=None):
        self.name = name
        self.email = email
        self.bio = bio
        self.password = bcrypt.hashpw(password.encode("utf8"), bcrypt.gensalt()).decode("utf8")

    def check_password(self, password):
        return bcrypt.checkpw(password.encode("utf8"), self.password.encode("utf8"))

with app.app_context():
    db.create_all()

# --- 3. Helper Functions ---

# Weather Helper
def get_weather_data(api_key, location, start_date, end_date):
    if not api_key: return None
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}?unitGroup=metric&include=days&key={api_key}&contentType=json"
    try:
        response = requests.get(base_url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None

# Chatbot Helper Functions
lemmatizer = WordNetLemmatizer()
# Load chatbot files safely
try:
    chat_model = keras.models.load_model('chat_model.h5')
    with open('words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pickle', 'rb') as f:
        classes = pickle.load(f)
    
    # Load Knowledge Base
    knowledge_base = []
    try:
        with open('intents.json') as file:
            intents_data = json.load(file)['intents']
            knowledge_base.extend(intents_data)
        with open('csv.json') as file:
            csv_data = json.load(file)
            for item in csv_data:
                knowledge_base.append({
                    "tag": item['cleaned_query'].lower().replace(' ', '_') + "_info",
                    "patterns": [item['cleaned_query']],
                    "responses": [item['response']]
                })
    except Exception as e:
        print(f"Error loading JSON files: {e}")

except Exception as e:
    print(f"Chatbot AI models not found. Chatbot will not work. Error: {e}")
    chat_model = None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if not chat_model: return None
    bow = bag_of_words(sentence, words)
    res = chat_model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return classes[results[0][0]]
    return None


# ==========================================
#               API ROUTES
# ==========================================

# --- 1. Itinerary Generation Route ---
@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary_api():
    # Get JSON data from React
    data = request.get_json()
    
    source = data.get("source")
    destination = data.get("destination")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    budget = data.get("budget")
    interests = data.get("interests", [])
    
    # Safe integer conversion
    try:
        adults = int(data.get("adults", 1))
        children = int(data.get("children", 0))
    except:
        adults = 1
        children = 0

    if not all([source, destination, start_date, end_date]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    # Get Weather
    weather_data = get_weather_data(WEATHER_API_KEY, destination, start_date, end_date)
    if not weather_data:
        weather_data = {"resolvedAddress": destination, "days": []}

    # Generate Itinerary using Bard/Gemini
    try:
        plan_markdown = bard.generate_itinerary(
            source=source,
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            adults=adults,
            children=children,
            budget=budget,
            interests=interests
        )
        # Convert markdown to HTML for display
        plan_html = mistune.html(plan_markdown)

        return jsonify({
            "status": "success",
            "plan_html": plan_html,
            "weather_data": weather_data
        })
    except Exception as e:
        print(f"Generation Error: {e}")
        return jsonify({"status": "error", "message": "Failed to generate itinerary"}), 500


# --- 2. Chatbot Route ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"reply": "Please send a message."})

    # 1. Predict Intent
    predicted_tag = predict_class(user_message)
    bot_response = ""

    # 2. Logic to find response
    if predicted_tag:
        # If the intent is asking for a recommendation, use Gemini AI
        if predicted_tag == 'destination_recommendation':
             # Ensure bard.py has get_chat_recommendations function
             try:
                 bot_response = bard.get_chat_recommendations(user_message)
             except AttributeError:
                 bot_response = "I recommend visiting the city center!"
        else:
            # Otherwise search the local knowledge base
            for intent in knowledge_base:
                if intent['tag'] == predicted_tag:
                    bot_response = random.choice(intent['responses'])
                    break
    
    if not bot_response:
        bot_response = "Sorry, I don't quite understand. Could you please rephrase?"

    return jsonify({"reply": bot_response})


# --- 3. Auth Routes (Login/Register) ---
@app.route("/api/users/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    bio = data.get("bio", "")

    if not all([name, email, password]):
        return jsonify({"message": "Please fill all fields"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"message": "User already exists"}), 400

    try:
        new_user = User(name=name, email=email, password=password, bio=bio)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({
            "id": new_user.id,
            "name": new_user.name,
            "email": new_user.email,
            "bio": new_user.bio
        }), 201
    except Exception as e:
        print(e)
        return jsonify({"message": "Server Error"}), 500

@app.route("/api/users/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session["user_id"] = user.id
        return jsonify({
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "bio": user.bio
        }), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401

# --- Server Start ---
if __name__ == "__main__":
    # Use Render's PORT environment variable, or default to 5002 locally
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=True)