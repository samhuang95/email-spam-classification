import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
import importlib.util
from pathlib import Path as _Path


# Try to load the explainability helper module from src/explainability.py
def _load_explainability_module():
    try:
        # prefer package import if available
        from src import explainability as expl
        return expl
    except Exception:
        p = _Path('src') / 'explainability.py'
        if p.exists():
            spec = importlib.util.spec_from_file_location('explainability', str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


_EXPLAIN = _load_explainability_module()


def predict_text_via_api(text: str, api_url: str = 'http://127.0.0.1:5000/predict'):
    try:
        r = requests.post(api_url, json={'text': text}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def predict_text_local(text: str):
    """Try to predict locally by loading a pipeline from artifacts/ or models/. Returns same dict as API."""
    import joblib
    from pathlib import Path

    # try to use explainability loader if available
    pipeline = None
    if _EXPLAIN is not None:
        try:
            pipeline = _EXPLAIN.load_pipeline_from_paths()
        except Exception:
            pipeline = None

    # fallback to common artifact path
    if pipeline is None:
        p = Path('artifacts/baseline_model.joblib')
        if p.exists():
            pipeline = joblib.load(p)
        else:
            # try models/ directory
            mdir = Path('models')
            if mdir.exists():
                for fname in ('spam_pipeline.joblib','pipeline.joblib','baseline_model.joblib'):
                    f = mdir / fname
                    if f.exists():
                        pipeline = joblib.load(f)
                        break

    if pipeline is None:
        raise FileNotFoundError('No local pipeline found in artifacts/ or models/')

    # perform prediction
    try:
        pred = pipeline.predict([text])[0]
    except Exception as e:
        raise RuntimeError(f'Local prediction failed: {e}')
    prob = None
    try:
        probs = pipeline.predict_proba([text])[0]
        # assume binary and class 1 is spam
        if len(probs) >= 2:
            prob = float(probs[1])
        else:
            prob = float(probs[0])
    except Exception:
        prob = None
    return {'label': 'spam' if int(pred) == 1 else 'ham', 'probability': prob}


def main():
    st.title('Spam Classifier (Baseline)')
    st.write('Upload dataset or input text to classify.')
    tabs = st.tabs(['Predict', 'Explore Data', 'Evaluation'])

    # Predict tab
    with tabs[0]:
        st.subheader('Single-message prediction')
        sample_messages = [
            'Free entry in 2 a wkly comp to win FA Cup final tkts',
            'Can we reschedule our meeting to next week?',
            'Win a £1000 cash prize! Call now',
            'Reminder: your appointment is tomorrow at 10am',
        ]
        sel = st.selectbox('Pick a sample message (or enter your own)', ['(none)'] + sample_messages)
        text = st.text_area('Input email / message text to classify', value=(sel if sel != '(none)' else ''))
        if st.button('Classify'):
            if not text:
                st.warning('Please enter text to classify.')
            else:
                # Try local prediction first (no need to run Flask API). Fallback to API if no local model.
                res = None
                try:
                    res = predict_text_local(text)
                except FileNotFoundError:
                    # local model not found, fallback to API
                    res = predict_text_via_api(text)
                except Exception as e:
                    # other local error; report and still try API
                    st.warning(f'Local prediction failed: {e} — trying remote API')
                    res = predict_text_via_api(text)

                if res is None:
                    st.error('No prediction result (local and remote attempts failed).')
                elif res.get('error'):
                    st.error(f"Error: {res['error']}")
                else:
                    st.metric('Prediction', res['label'])
                    if res.get('probability') is not None:
                        st.progress(min(max(res['probability'], 0.0), 1.0))
                        st.write(f"Spam probability: {res['probability']:.3f}")
                    # Explainability: allow local explanations when a local model exists
                    show_explain = st.checkbox('Show local explanation (requires local model in artifacts/ or models/)')
                    if show_explain:
                        if _EXPLAIN is None:
                            st.error('Explainability module not found (src/explainability.py).')
                        else:
                            try:
                                pipeline = None
                                try:
                                    pipeline = _EXPLAIN.load_pipeline_from_paths()
                                except Exception as e:
                                    st.warning(f'Could not auto-load pipeline: {e}')
                                    pipeline = None

                                if pipeline is None:
                                    st.info('No local pipeline found. Place a pipeline in `models/` or `artifacts/baseline_model.joblib`.')
                                else:
                                    with st.spinner('Computing token importance...'):
                                        toks = _EXPLAIN.token_importance_by_removal(text, pipeline, top_k=15)
                                    if toks:
                                        df_t = pd.DataFrame(toks, columns=['token', 'delta_prob'])
                                        st.write('Token importance (delta in spam probability when token removed):')
                                        st.dataframe(df_t)
                                        try:
                                            st.bar_chart(df_t.set_index('token')['delta_prob'].abs())
                                        except Exception:
                                            pass

                                    # Try SHAP explanation if available
                                    shap_res = None
                                    try:
                                        shap_res = _EXPLAIN.try_shap_explain([text], pipeline)
                                    except Exception:
                                        shap_res = None

                                    if shap_res is None:
                                        st.info('SHAP not available or failed; token-removal results shown above.')
                                    else:
                                        st.write('SHAP returned results (raw arrays):')
                                        st.write(shap_res)
                            except Exception as e:
                                st.error(f'Explainability computation failed: {e}')

                        # Batch prediction UI
                        st.markdown('---')
                        st.subheader('Batch prediction (CSV)')
                        st.write('Upload a CSV with a `text` column or two columns (label,text). The app will predict labels and probabilities.')
                        batch_file = st.file_uploader('Upload CSV for batch prediction', type=['csv'], key='batch')
                        if batch_file is not None:
                            try:
                                bdf = pd.read_csv(batch_file, header=0)
                            except Exception:
                                # try no header
                                bdf = pd.read_csv(batch_file, header=None).rename(columns={0: 'label', 1: 'text'})
                            # try to find text column
                            if 'text' not in bdf.columns:
                                # try second column
                                if len(bdf.columns) >= 2:
                                    bdf = bdf.rename(columns={bdf.columns[1]: 'text'})
                                else:
                                    st.error('Could not find a `text` column in the uploaded CSV.')
                                    bdf = None
                            if bdf is not None:
                                texts = bdf['text'].astype(str).tolist()
                                preds = []
                                for t in texts:
                                    try:
                                        r = predict_text_local(t)
                                    except FileNotFoundError:
                                        r = predict_text_via_api(t)
                                    except Exception as e:
                                        r = {'error': str(e)}
                                    preds.append(r)
                                pred_df = pd.DataFrame(preds)
                                out = pd.concat([bdf.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
                                st.write('Predictions:')
                                st.dataframe(out.head(200))
                                # provide download
                                csv_bytes = out.to_csv(index=False).encode('utf-8')
                                st.download_button('Download predictions as CSV', data=csv_bytes, file_name='predictions.csv', mime='text/csv')

    # Explore Data tab
    with tabs[1]:
        st.subheader('Dataset preview & distributions')
        uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
        visuals_path = Path('artifacts/visuals.json')
        if uploaded is not None:
            df = pd.read_csv(uploaded, header=None).rename(columns={0: 'label', 1: 'text'})
            st.write(df.head())
            dist = df['label'].value_counts().to_dict()
            st.bar_chart(pd.Series(dist))
        elif visuals_path.exists():
            v = json.loads(visuals_path.read_text())
            st.write('Label distribution:')
            st.bar_chart(pd.Series(v['label_distribution']))
            st.write('Top words (spam):')
            spam_top = v['top_words']['spam'][:20]
            st.dataframe(pd.DataFrame(spam_top, columns=['word', 'count']))
            st.write('Top words (ham):')
            ham_top = v['top_words']['ham'][:20]
            st.dataframe(pd.DataFrame(ham_top, columns=['word', 'count']))
        else:
            st.info('No visuals found. Run the demo generator to produce visualization data.')

    # Evaluation tab
    with tabs[2]:
        st.subheader('Evaluation metrics & curves')
        visuals_path = Path('artifacts/visuals.json')
        if visuals_path.exists():
            v = json.loads(visuals_path.read_text())
            metrics = None
            # load metrics.json if present
            mpath = Path('artifacts/metrics.json')
            if mpath.exists():
                metrics = json.loads(mpath.read_text())
            if metrics:
                st.metric('Precision', metrics.get('precision'))
                st.metric('Recall', metrics.get('recall'))
                st.metric('F1', metrics.get('f1'))
                st.metric('ROC AUC', metrics.get('roc_auc'))
            # ROC curve
            roc = v.get('curves', {}).get('roc', {})
            pr = v.get('curves', {}).get('pr', {})
            import plotly.express as px
            import pandas as _pd

            if roc and roc.get('fpr'):
                df_roc = _pd.DataFrame({'fpr': roc['fpr'], 'tpr': roc['tpr']})
                fig = px.line(df_roc, x='fpr', y='tpr', title='ROC Curve')
                st.plotly_chart(fig)
            if pr and pr.get('precision'):
                df_pr = _pd.DataFrame({'precision': pr['precision'], 'recall': pr['recall']})
                fig2 = px.line(df_pr, x='recall', y='precision', title='Precision-Recall Curve')
                st.plotly_chart(fig2)
            # confusion matrix
            cm = v.get('curves', {}).get('confusion_matrix')
            if cm:
                cm_df = _pd.DataFrame([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]], index=['actual_ham', 'actual_spam'], columns=['pred_ham', 'pred_spam'])
                st.write('Confusion matrix:')
                st.dataframe(cm_df)
        else:
            st.info('No evaluation visuals found. Run the demo generator to produce visualization data.')


if __name__ == '__main__':
    main()
