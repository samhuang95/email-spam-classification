from typing import List, Dict
import numpy as np
import shap


def explain_with_shap(pipeline, text: str, background_texts: List[str], top_n: int = 20) -> Dict:
    """Explain a single text using SHAP KernelExplainer on TF-IDF features.

    Returns top_n features and their SHAP values.
    """
    # Expect pipeline: TfidfVectorizer -> classifier (supports predict_proba on numeric arrays)
    # Extract steps
    try:
        vectorizer = pipeline.named_steps['tfidfvectorizer']
    except Exception:
        # fallback: find first transformer with transform
        vectorizer = None
        for name, step in pipeline.named_steps.items():
            if hasattr(step, 'transform') and hasattr(step, 'get_feature_names_out'):
                vectorizer = step
                break
    if vectorizer is None:
        raise RuntimeError('Could not find vectorizer in pipeline')
    # classifier is last step
    classifier = pipeline.steps[-1][1]

    # Prepare background
    bg = vectorizer.transform(background_texts)
    try:
        bg = bg.toarray()
    except Exception:
        bg = np.array(bg)

    def f(x):
        # x is numeric array matching tf-idf features
        return classifier.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(f, bg)
    x = vectorizer.transform([text])
    try:
        x_arr = x.toarray()
    except Exception:
        x_arr = np.array(x)
    shap_values = explainer.shap_values(x_arr, nsamples=100)
    # shap_values shape: (n_features,) or (1, n_features)
    sv = np.array(shap_values)
    if sv.ndim == 2:
        sv = sv[0]
    # map indices to feature names
    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    top = [{'feature': feature_names[i], 'shap_value': float(sv[i])} for i in idx]
    return {'base_value': float(explainer.expected_value), 'top_features': top}
