from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('linear_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    size = data['size']
    prediction = model.predict(pd.DataFrame({'size': [size]}))
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)