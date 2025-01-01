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

# Load English model and tokenizer
try:
    en_model = load_model(r"C:\Users\lataeq\Code\RNN-ML\sentiment_analysis_model.h5")
    with open(r"C:\Users\lataeq\Code\RNN-ML\data\tokenizer.pkl", "rb") as en_handle:
        en_tokenizer = pickle.load(en_handle)
except Exception as e:
    raise RuntimeError(f"Error loading English model or tokenizer: {str(e)}")

# Load Thai model and tokenizer
try:
    th_model = load_model(r"C:\Users\lataeq\Code\rnn\RNN-ML\models\model.keras")
    with open(r"C:\Users\lataeq\Code\rnn\RNN-ML\models\tokenizer.pkl", "rb") as th_handle:
        th_tokenizer = pickle.load(th_handle)
    th_word_index = th_tokenizer.word_index  # Use word_index for Thai preprocessing
except Exception as e:
    raise RuntimeError(f"Error loading Thai model or tokenizer: {str(e)}")

# Constants
MAX_LENGTH = 32

# Preprocessing function for English input
def preprocess_english_text(text):
    try:
        tokenized_sentence = en_tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(tokenized_sentence, maxlen=MAX_LENGTH, padding="pre")
        return padded_sequence
    except Exception as e:
        raise ValueError(f"Error in preprocessing English text: {str(e)}")

# Preprocessing function for Thai input
def preprocess_thai_text(text):
    try:
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        sequence = [th_word_index.get(word, 0) for word in tokens]
        padded_sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding="post")
        return padded_sequence
    except Exception as e:
        raise ValueError(f"Error in preprocessing Thai text: {str(e)}")

# Sentiment analysis function
def analyze_sentiment(text, language):
    try:
        if language == "en":
            input_sequence = preprocess_english_text(text)
            predictions = en_model.predict(input_sequence)
        elif language == "th":
            input_sequence = preprocess_thai_text(text)
            predictions = th_model.predict(input_sequence)
        else:
            return {"error": "Unsupported language"}

        # Determine sentiment
        prediction = predictions.argmax()
        confidence = {
            "Negative": round(predictions[0][2] * 100, 2),
            "Neutral": round(predictions[0][1] * 100, 2),
            "Positive": round(predictions[0][0] * 100, 2),
        }
        sentiment = ["Positive", "Neutral", "Negative"][prediction]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}

# Routes
@app.route("/")
def index():
    return render_template("index.html")  # Adjust index.html to include language selection

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "")
        language = request.form.get("language", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400
        if language not in ["en", "th"]:
            return jsonify({"error": "Unsupported or missing language"}), 400

        # Perform sentiment analysis
        result = analyze_sentiment(text, language)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
