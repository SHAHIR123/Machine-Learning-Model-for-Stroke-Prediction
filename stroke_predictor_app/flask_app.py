import json
import os
from flask import Flask,jsonify,request, render_template
from flask_cors import CORS
from predictor import stroke_predictor

app = Flask(__name__)
CORS(app)



@app.route("/", methods=['GET'])

def default():
  return render_template('index.html')

@app.route("/stroke/",methods=['GET'])
def return_stroke():
    age = request.args.get('age')
    hypertension = request.args.get('hypertension')
    heart_disease = request.args.get('heart_disease')
    avg_glucose_level = request.args.get('avg_glucose_level')
    bmi = request.args.get('bmi')
    stroke = stroke_predictor().predict(age, hypertension, heart_disease, avg_glucose_level, bmi) 
    if stroke==0:
        return("Low Stroke Possibility")
    else:
        return("High Stroke Possibility")
   

if __name__ == "__main__":
    app.run() 


