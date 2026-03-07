import joblib, json, os, numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

risk_model = joblib.load(os.path.join(MODEL_DIR, 'risk_progression_model.pkl'))
scaler_cls = joblib.load(os.path.join(MODEL_DIR, 'scaler_classification.pkl'))
re_model   = joblib.load(os.path.join(MODEL_DIR, 'has_re_model_improved.pkl'))
re_scaler  = joblib.load(os.path.join(MODEL_DIR, 'has_re_scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'has_re_features.json')) as f:
    re_meta = json.load(f)
with open(os.path.join(MODEL_DIR, 'feature_columns.json')) as f:
    feat_cols = json.load(f)

print('Stage 1 features:', len(re_meta['feature_columns']))
print('Stage 2 features:', len(feat_cols))
print('Stage 1 model:', type(re_model).__name__, 'AUC:', re_meta['metrics']['auc'])
print('Stage 2 model:', type(risk_model).__name__)
print('scaler_cls n_features:', scaler_cls.n_features_in_)
print('Stage 2 features match scaler:', scaler_cls.n_features_in_ == len(feat_cols))

with open(os.path.join(MODEL_DIR, 'model_metadata.json')) as f:
    meta = json.load(f)

print()
print('=== Model Performance Summary ===')
print('Stage 1:', meta['re_model']['name'], ' AUC=', round(meta['re_model']['auc'],4))
print('Stage 2:', meta['risk_model']['name'], ' AUC=', round(meta['risk_model']['auc'],4))
print('Stage 3:', meta['diopter_model']['name'], ' MAE=', round(meta['diopter_model']['mae'],3), ' R2=', round(meta['diopter_model']['r2'],4))
print()
print('Comparison:')
for stage, vals in meta['comparison'].items():
    print(' ', stage, ':', vals)
