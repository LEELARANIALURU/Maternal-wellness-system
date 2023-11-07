from flask import Flask, request, jsonify
from flask_cors import CORS
import util

application = Flask(__name__)
CORS(application)

@application.route('/predict_risk', methods=['GET', 'POST'])
def predict_home_price():
    age = int(request.form['age'])
    sbp = int(request.form['sbp'])
    dbp = int(request.form['dbp'])
    sugar = int(request.form['sugar'])
    temp = int(request.form['temp'])
    heart = int(request.form['heart'])
    

    response = jsonify({
        'estRisk': util.get_estimated_risk(age, sbp, dbp, sugar, temp, heart)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Maternity Health Risk Prediction...")
    util.load_saved_artifacts()
    application.run()