from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import util

application = Flask(__name__, static_folder=r'C:\Users\hp\OneDrive\Desktop\Maternal-wellness-system\client', template_folder=r'C:\Users\hp\OneDrive\Desktop\Maternal-wellness-system\client')
application.config['EXPLAIN_TEMPLATE_LOADING'] = True
application.config['EXPLAIN_STATIC_LOADING'] = True
CORS(application)

@application.route('/predict_risk', methods=['GET', 'POST'])
def predict_maternal_risk():
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
        return render_template('maternal.html')
    
@application.route('/fetal_risk', methods=['GET', 'POST'])
def predict_fetal_risk():
    if request.method=='POST':
        baseline = float(request.form['baseline'])
        accelerations = float(request.form['accelerations'])
        fetal_movement = float(request.form['fetal_movement'])
        uterine_contractions = float(request.form['uterine_contractions'])
        light_decelerations = float(request.form['light_decelerations'])
        severe_decelerations = float(request.form['severe_decelerations'])
        prolongued_decelerations = float(request.form['prolongued_decelerations'])
        abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
        mean_value_of_short_term_variability = float(request.form['mean_value_of_short_term_variability'])
        percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        mean_value_of_long_term_variability = float(request.form['mean_value_of_long_term_variability'])
        histogram_width = float(request.form['histogram_width'])
        histogram_min = float(request.form['histogram_min'])
        histogram_max = float(request.form['histogram_max'])
        histogram_number_of_peaks = float(request.form['histogram_number_of_peaks'])
        histogram_mode = float(request.form['histogram_mode'])
        histogram_mean = float(request.form['histogram_mean'])
        histogram_median = float(request.form['histogram_median'])
        histogram_variance = float(request.form['histogram_variance'])
        histogram_tendency = float(request.form['histogram_tendency'])
    

        response = jsonify({
            'fetalRisk': int(util.get_fetal_risk(baseline, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations,
                                                 prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability,
                                                 percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability,
                                                 histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_mode, histogram_mean,
                                                 histogram_median, histogram_variance, histogram_tendency))
        })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    else:
        return render_template(r'fetal.html')


if __name__ == "__main__":
    print("Starting Python Flask Server For Maternity Health Risk Prediction...")
    util.load_saved_artifacts()
    application.run(debug=True)