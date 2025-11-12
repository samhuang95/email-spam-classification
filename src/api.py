from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)


def load_model(path='artifacts/baseline_model.joblib'):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


MODEL = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    data = request.get_json() or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    pred = MODEL.predict([text])[0]
    prob = None
    try:
        prob = MODEL.predict_proba([text])[0][1]
    except Exception:
        pass
    return jsonify({'label': 'spam' if int(pred) == 1 else 'ham', 'probability': float(prob) if prob is not None else None})


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    data = request.get_json() or {}
    texts = data.get('texts')
    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'No texts provided (expecting JSON {"texts": [...]})'}), 400
    preds = []
    for t in texts:
        p = MODEL.predict([t])[0]
        prob = None
        try:
            prob = MODEL.predict_proba([t])[0][1]
        except Exception:
            pass
        preds.append({'label': 'spam' if int(p) == 1 else 'ham', 'probability': float(prob) if prob is not None else None})
    return jsonify({'predictions': preds})


@app.route('/explain', methods=['POST'])
def explain():
    """Simple token-removal explanation: measure probability change when removing tokens."""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    data = request.get_json() or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    base_prob = None
    try:
        base_prob = MODEL.predict_proba([text])[0][1]
    except Exception:
        try:
            base_prob = float(MODEL.decision_function([text])[0])
        except Exception:
            base_prob = None
    import re
    tokens = re.findall(r"\w+", text)
    impacts = []
    for tok in set(tokens):
        modified = ' '.join([t for t in tokens if t != tok])
        try:
            p = MODEL.predict_proba([modified])[0][1]
            impacts.append({'token': tok, 'delta': float(base_prob - p)})
        except Exception:
            continue
    impacts = sorted(impacts, key=lambda x: abs(x['delta']), reverse=True)[:20]
    return jsonify({'base_prob': base_prob, 'impacts': impacts})


@app.route('/explain_shap', methods=['POST'])
def explain_shap():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    data = request.get_json() or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    # load some background texts from data file if available
    import pandas as pd
    bg = []
    try:
        df = pd.read_csv('data/sms_spam.csv', header=None).rename(columns={0:'label',1:'text'})
        bg = df['text'].astype(str).sample(min(50, len(df))).tolist()
    except Exception:
        bg = [text]
    try:
        from src.shap_explain import explain_with_shap
        res = explain_with_shap(MODEL, text, bg, top_n=20)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
