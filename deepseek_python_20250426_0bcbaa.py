from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model (works on Render's cloud)
toxicity = pipeline(
    "text-classification",
    model="unitary/toxic-bert",  # Or "distilbert-base-uncased-finetuned-sst-2-english" for lighter weight
    device=-1  # Uses CPU (remove if you upgrade to paid GPU)
)

@app.route("/check", methods=["POST"])
def check_comment():
    comment = request.json.get("comment")
    result = toxicity(comment)[0]
    return jsonify({"label": result['label'], "score": result['score']})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render uses port 10000