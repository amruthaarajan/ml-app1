from flask import Flask, request, abort, jsonify
import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)

model = NLPModel()

clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

@app.route('/sentimentalanalysis', methods=['POST'])
def predictSentiment():
    if not request.data:
        abort(400)
    user_query = request.data
    uq_vectorized = model.vectorizer_transform(np.array([user_query]))
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)

    # Output either 'Negative' or 'Positive' along with the score
    if prediction == 0:
        pred_text = 'Negative'
    else:
        pred_text = 'Positive'

    # round the predict proba value and set to new variable
    confidence = round(pred_proba[0], 3)

    # create JSON object
    output = {'prediction': pred_text, 'confidence': confidence}

    return jsonify(output), 200

if __name__ == '__main__':
    app.run(debug=True)
