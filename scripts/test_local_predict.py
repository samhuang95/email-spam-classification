from pathlib import Path
import joblib
p=Path('artifacts/baseline_model.joblib')
if not p.exists():
    print('MODEL_NOT_FOUND')
else:
    model=joblib.load(p)
    text='Free entry in 2 a wkly comp to win FA Cup final tkts'
    try:
        pred=model.predict([text])[0]
        prob=None
        try:
            prob=model.predict_proba([text])[0][1]
        except Exception:
            prob=None
        print('PRED',pred,'PROB',prob)
    except Exception as e:
        print('PRED_ERR',e)
