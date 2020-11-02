import numpy as np
from flask import Flask, render_template, redirect, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        
        int_features = np.array([[int(x) for x in request.form.values()]])
        prediction = model.predict(int_features)
        
        
    
        output = round(prediction[0], 2)
        return render_template('index.html',prediction_text= (output))
           
        
    
if __name__ == '__main__':
    app.run(debug=True)
    
											