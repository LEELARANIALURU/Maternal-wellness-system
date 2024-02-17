from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import util

application = Flask(__name__, template_folder='D:\Documents\Maternal-wellness-system\client')
application.config['EXPLAIN_TEMPLATE_LOADING'] = True
CORS(application)

@application.route('/predict_risk', methods=['GET', 'POST'])
def predict_home_price():
    if request.method=='POST':
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
    else:
        return render_template('index.html')


if __name__ == "__main__":
    print("Starting Python Flask Server For Maternity Health Risk Prediction...")
    util.load_saved_artifacts()
    application.run(debug=True)