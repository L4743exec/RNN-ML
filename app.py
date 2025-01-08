from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize
import pickle

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # จำเป็นสำหรับ CSRF

# Enable CORS
CORS(app)

# Enable CSRF protection
csrf = CSRFProtect(app)

# Enable Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],  # สามารถปรับแต่ง rate limit ได้
)

# Load English model and tokenizer
try:
    en_model = load_model(r"models/en/sentiment_analysis_model.h5")
    with open(r"models/en/tokenizer.pkl", "rb") as en_handle:
        en_tokenizer = pickle.load(en_handle)
except Exception as e:
    raise RuntimeError(f"Error loading English model or tokenizer: {str(e)}")

# Load Thai model and tokenizer
try:
    th_model = load_model(r"models/th/model.keras")
    with open(r"models/th/tokenizer.pkl", "rb") as th_handle:
        th_tokenizer = pickle.load(th_handle)
    # Use word_index for Thai preprocessing
    th_word_index = th_tokenizer.word_index
except Exception as e:
    raise RuntimeError(f"Error loading Thai model or tokenizer: {str(e)}")

# Constants
MAX_LENGTH = 32

# Preprocessing function for English input
def preprocess_english_text(text):
    try:
        tokenized_sentence = en_tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(
            tokenized_sentence, maxlen=MAX_LENGTH, padding="pre")
        print(get_remote_address(),text, padded_sequence)
        return padded_sequence
    except Exception as e:
        raise ValueError(f"Error in preprocessing English text: {str(e)}")

# Preprocessing function for Thai input
def preprocess_thai_text(text):
    try:
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        sequence = [th_word_index.get(word, 0) for word in tokens]
        padded_sequence = pad_sequences(
            [sequence], maxlen=MAX_LENGTH, padding="post")
        print(text, padded_sequence)
        return padded_sequence
    except Exception as e:
        raise ValueError(f"Error in preprocessing Thai text: {str(e)}")

# Sentiment analysis function
def analyze_sentiment(text, language):
    try:
        if language == "en":
            input_sequence = preprocess_english_text(text)
            predictions = en_model.predict(input_sequence)
            prediction = predictions.argmax()
            confidence = {
                "Negative": round(predictions[0][0] * 100, 2),
                "Neutral": round(predictions[0][1] * 100, 2),
                "Positive": round(predictions[0][2] * 100, 2),
            }
            sentiment = ["Negative", "Neutral", "Positive"][prediction]
        elif language == "th":
            input_sequence = preprocess_thai_text(text)
            predictions = th_model.predict(input_sequence)
            prediction = predictions.argmax()
            confidence = {
                "Negative": round(predictions[0][1] * 100, 2),
                "Neutral": round(predictions[0][0] * 100, 2),
                "Positive": round(predictions[0][2] * 100, 2),
            }
            sentiment = ["Neutral", "Negative", "Positive"][prediction]
        else:
            return {"error": "Unsupported language"}
        # Determine sentiment
        return {
            "sentiment": sentiment,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}

# Routes
@app.route("/")
@limiter.limit("10 per minute")  # Apply rate limit to this route
def index():
    # Adjust index.html to include language selection
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@csrf.exempt  # Disable CSRF for this route if using external clients
@limiter.limit("5 per second")  # Apply rate limit to /predict
def predict():
    try:
        text = request.form.get("text", "")
        language = request.form.get("language", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400
        if language not in ["en", "th"]:
            language = "en"

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
