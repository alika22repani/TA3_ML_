import joblib, pandas as pd
from flask import Flask, render_template, request
from features import build_features

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
THRESHOLD = 0.5

FIELDS = [('air_temp', 'Air temperature [K]', 300.0), ('process_temp', 'Process temperature [K]', 310.0),
          ('rpm', 'Rotational speed [rpm]', 1500), ('torque', 'Torque [Nm]', 40.0),
          ('tool_wear', 'Tool wear [min]', 100)]

@app.route('/', methods=['GET', 'POST'])
def index():
    result, error, form = None, None, {}
    if request.method == 'POST':
        form = request.form.to_dict()
        try:
            row = {k: float(form[k]) for k, _, _ in FIELDS}
            row['type'] = form.get('type', 'L')
            X = scaler.transform(build_features(pd.DataFrame([row])))
            p = float(model.predict_proba(X)[0, 1])
            result = {'label': 'Berisiko Gagal' if p >= THRESHOLD else 'Tidak Berisiko',
                      'risk': p >= THRESHOLD, 'prob': round(p * 100, 2)}
        except (ValueError, KeyError):
            error = 'Semua kolom harus diisi dengan angka yang valid.'
    return render_template('index.html', fields=FIELDS, result=result, error=error, form=form)

if __name__ == '__main__':
    app.run(debug=True)
