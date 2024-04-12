from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

loaded_model = joblib.load('xgb_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
   
    return render_template('predict.html')

@app.route("/doctorslist", methods=['GET', 'POST'])
def doctorsList():
   
    return render_template('doctorsList.html')

@app.route("/result", methods=['GET', 'POST'])
def result():
    drug = int(request.form['drug'])
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    ascites = int(request.form['ascites'])
    hepatomegaly = int(request.form['hepatomegaly'])
    spiders = int(request.form['spiders'])
    edema = int(request.form['edema'])
    bilirubin = float(request.form['bilirubin'])
    cholesterol = float(request.form['cholesterol'])
    albumin = float(request.form['albumin'])   
    copper = float(request.form['copper'])
    alk_phos = float(request.form['alk_phos'])
    sgot = float(request.form['sgot'])
    tryglicerides = float(request.form['tryglicerides'])
    platelets = float(request.form['platelets'])
    prothrombin = float(request.form['prothrombin'])

    input_data = np.array([drug,age,gender,ascites,hepatomegaly,spiders,edema,bilirubin, cholesterol, albumin, copper, alk_phos, sgot, tryglicerides, platelets, prothrombin])
    #print(input_data)
    predictions = loaded_model.predict([input_data])
    #print(predictions)
    return render_template('result.html',predictions=predictions+1)
      

if __name__ == "__main__":
    app.run(debug=True)
