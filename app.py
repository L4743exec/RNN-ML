from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model = load_model("sentiment_analysis_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LENGTH = 32

def SentimentAnalysis(text):
    try:
        sentence = [text]
        tokenized_sentence = tokenizer.texts_to_sequences(sentence)
        input_sequence = pad_sequences(tokenized_sentence, maxlen=MAX_LENGTH, padding="pre")
        prediction_ = model.predict(input_sequence)
        prediction = prediction_.argmax()
        
        # Confidence calculation
        confidence = {
            "Negative": round(prediction_[0][0] * 100, 2),
            "Neutral": round(prediction_[0][1] * 100, 2),
            "Positive": round(prediction_[0][2] * 100, 2),
        }

        # Sentiment assignment
        sentiment = ["Negative", "Neutral", "Positive"][prediction]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Use the SentimentAnalysis function
        result = SentimentAnalysis(text)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
