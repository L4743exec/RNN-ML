from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # สำหรับจัดการ CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
CORS(app)  # เปิดใช้งาน CORS ทั่วทั้งแอป

# โหลดโมเดลและ Tokenizer
model = load_model("sentiment_analysis_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LENGTH = 32  # ใช้ขนาดที่ตรงกับโมเดล

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ดึงข้อความที่ส่งมา
        text = request.form.get("text", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # แปลงข้อความเป็นลำดับ
        tokenized_sentence = tokenizer.texts_to_sequences([text])
        input_sequence = pad_sequences(tokenized_sentence, maxlen=MAX_LENGTH, padding="pre")

        # ทำนายผลลัพธ์
        prediction_ = model.predict(input_sequence).tolist()[0]
        prediction = int(np.argmax(prediction_))

        # แปลงผลลัพธ์เป็นข้อความ
        sentiment = ["Negative", "Neutral", "Positive"]
        result = {
            "sentiment": sentiment[prediction],
            "confidence": {
                "Negative": round(float(prediction_[0]) * 100, 2),
                "Neutral": round(float(prediction_[1]) * 100, 2),
                "Positive": round(float(prediction_[2]) * 100, 2),
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
