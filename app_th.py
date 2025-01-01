from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
try:
    model = load_model(r"C:\Users\lataeq\Code\rnn\RNN-ML\models\model.keras")  # Updated model file name
    with open(r"C:\Users\lataeq\Code\rnn\RNN-ML\models\tokenizer.pkl", "rb") as handle:
        train_word_tokenizer = pickle.load(handle)  # Renamed for clarity
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {str(e)}")

# Constants
word_index = train_word_tokenizer.word_index  # Match training tokenizer
MAX_LENGTH = 32  # Match training max length

# Preprocessing function for Thai input
def preprocess_text(text, word_index, max_length):
    try:
        # Step 1: Tokenize the input text
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        
        # Step 2: Convert tokens to sequences using the training tokenizer's word_index
        sequence = [word_index.get(word, 0) for word in tokens]
        
        # Step 3: Pad the sequence to match the max_length used in training
        padded_sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        return padded_sequence
    except Exception as e:
        raise ValueError(f"Error in preprocessing text: {str(e)}")

# Function for sentiment analysis
def SentimentAnalysis(text):
    try:
        # Preprocess the Thai input text
        input_sequence = preprocess_text(text, word_index, MAX_LENGTH)
        
        # Make prediction
        predictions = model.predict(input_sequence)
        prediction = predictions.argmax()
        
        # Confidence scores
        confidence = {
            "Negative": round(predictions[0][2] * 100, 2),  # Adjusted order for consistency
            "Neutral": round(predictions[0][1] * 100, 2),
            "Positive": round(predictions[0][0] * 100, 2),
        }

        # Assign sentiment label
        sentiment = ["Positive", "Neutral", "Negative"][prediction]  # Adjusted order for consistency

        return {
            "sentiment": sentiment,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get text input from request
        text = request.form.get("text", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Perform sentiment analysis
        result = SentimentAnalysis(text)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
